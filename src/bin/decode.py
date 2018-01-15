#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import argparse
import logging
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from torchvision import transforms

from utils import find_files, read_hdf5, read_txt
from wavenet import WaveNet, decode_mu_law, encode_mu_law


def decode_generator(feat_list, wav_transform=None,
                     feat_transform=None, use_speaker_code=False,
                     upsampling_factor=0):
    """DECODE BATCH GENERATOR

    Args:
        featdir (str): directory including feat files
        wav_transform (func): preprocessing function for waveform
        feat_transform (func): preprocessing function for aux feats
        use_speaker_code (bool): whether to use speaker code
        upsampling_factor (int): upsampling factor

    Return:
        (object): generator instance
    """
    # process over all of files
    for featfile in feat_list:
        x = np.zeros((1))
        if upsampling_factor == 0:
            h = read_hdf5(featfile, "/feat")
        else:
            h = read_hdf5(featfile, "/feat_org")
        if use_speaker_code:
            sc = read_hdf5(featfile, "/speaker_code")
            sc = np.tile(sc, [h.shape[0], 1])
            h = np.concatenate([h, sc], axis=1)

        # perform pre-processing
        if wav_transform is not None:
            x = wav_transform(x)
        if feat_transform is not None:
            h = feat_transform(h)

        x = x.unsqueeze(0)
        h = h.transpose(0, 1).unsqueeze(0)
        if upsampling_factor == 0:
            n_samples = h.size(2) - 1
        else:
            n_samples = h.size(2) * upsampling_factor - 1
        feat_id = os.path.basename(featfile).replace(".h5", "")

        yield feat_id, (x, h, n_samples)


def main():
    parser = argparse.ArgumentParser()
    # decode setting
    parser.add_argument("--feats", required=True,
                        type=str, help="list or directory of aux feat files")
    parser.add_argument("--stats", required=True,
                        type=str, help="hdf5 file including statistics")
    parser.add_argument("--checkpoint", required=True,
                        type=str, help="model file")
    parser.add_argument("--config", required=True,
                        type=str, help="configure file")
    parser.add_argument("--outdir", required=True,
                        type=str, help="directory to save generated samples")
    parser.add_argument("--fs", default=16000,
                        type=int, help="sampling rate")
    parser.add_argument("--n_jobs", default=5,
                        type=int, help="number of parallel jobs per gpu")
    parser.add_argument("--n_gpus", default=1,
                        type=int, help="number of gpus")
    # other setting
    parser.add_argument("--intervals", default=5000,
                        type=int, help="log interval")
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log level")
    args = parser.parse_args()

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # set log level
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # fix seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load config
    config = torch.load(args.config)

    # get file list
    if os.path.isdir(args.feats):
        feat_list = sorted(find_files(args.feats, "*.h5"))
    elif os.path.isfile(args.feats):
        feat_list = read_txt(args.feats)
    else:
        logging.error("--feats should be directory or list.")
        sys.exit(1)

    # prepare the file list for parallel decoding
    if args.n_gpus > 0:
        feat_lists = np.array_split(feat_list, args.n_jobs * args.n_gpus)
    else:
        feat_lists = np.array_split(feat_list, args.n_jobs)
    feat_lists = [f_list.tolist() for f_list in feat_lists]

    # define gpu decode function
    def gpu_decode(feat_list, gpu):
        with torch.cuda.device(gpu):
            # define model and load parameters
            model = WaveNet(
                n_quantize=config.n_quantize,
                n_aux=config.n_aux,
                n_resch=config.n_resch,
                n_skipch=config.n_skipch,
                dilation_depth=config.dilation_depth,
                dilation_repeat=config.dilation_repeat,
                kernel_size=config.kernel_size,
                upsampling_factor=config.upsampling_factor)
            model.load_state_dict(torch.load(args.checkpoint)["model"])
            model.eval()
            model.cuda()
            torch.backends.cudnn.benchmark = True

            # define transform
            scaler = StandardScaler()
            scaler.mean_ = read_hdf5(args.stats, "/mean")
            scaler.scale_ = read_hdf5(args.stats, "/scale")
            wav_transform = transforms.Compose([
                lambda x: encode_mu_law(x, config.n_quantize),
                lambda x: torch.from_numpy(x).long().cuda(),
                lambda x: Variable(x, volatile=True)])
            feat_transform = transforms.Compose([
                lambda x: scaler.transform(x),
                lambda x: torch.from_numpy(x).float().cuda(),
                lambda x: Variable(x, volatile=True)])

            # define generator
            generator = decode_generator(
                feat_list,
                wav_transform=wav_transform,
                feat_transform=feat_transform,
                use_speaker_code=config.use_speaker_code,
                upsampling_factor=config.upsampling_factor)

            # decode
            for feat_id, (x, h, n_samples) in generator:
                if os.path.exists(args.outdir + "/" + feat_id + ".wav"):
                    logging.info("%s already exists." % feat_id)
                else:
                    logging.info("decoding %s (length = %d)" % (feat_id, n_samples))
                    samples = model.fast_generate(x, h, n_samples, args.intervals)
                    wav = decode_mu_law(samples, config.n_quantize)
                    sf.write(args.outdir + "/" + feat_id + ".wav", wav, args.fs, "PCM_16")
                    logging.info("wrote %s.wav in %s." % (feat_id, args.outdir))

    # define cpu decode function
    def cpu_decode(feat_list):
        # define model and load parameters
        model = WaveNet(
            n_quantize=config.n_quantize,
            n_aux=config.n_aux,
            n_resch=config.n_resch,
            n_skipch=config.n_skipch,
            dilation_depth=config.dilation_depth,
            dilation_repeat=config.dilation_repeat,
            kernel_size=config.kernel_size,
            upsampling_factor=config.upsampling_factor)
        model.load_state_dict(torch.load(args.checkpoint)["model"])
        model.eval()
        model.cpu()

        # define transform
        scaler = StandardScaler()
        scaler.mean_ = read_hdf5(args.stats, "/mean")
        scaler.scale_ = read_hdf5(args.stats, "/scale")
        wav_transform = transforms.Compose([
            lambda x: encode_mu_law(x, config.n_quantize),
            lambda x: torch.from_numpy(x).long(),
            lambda x: Variable(x, volatile=True)])
        feat_transform = transforms.Compose([
            lambda x: scaler.transform(x),
            lambda x: torch.from_numpy(x).float(),
            lambda x: Variable(x, volatile=True)])

        # define generator
        generator = decode_generator(
            feat_list,
            wav_transform=wav_transform,
            feat_transform=feat_transform,
            use_speaker_code=config.use_speaker_code,
            upsampling_factor=config.upsampling_factor)

        # decode
        for feat_id, (x, h, n_samples) in generator:
            if os.path.exists(args.outdir + "/" + feat_id + ".wav"):
                logging.info("%s already exists." % feat_id)
            else:
                logging.info("decoding %s (length = %d)" % (feat_id, n_samples))
                samples = model.fast_generate(x, h, n_samples, args.intervals)
                wav = decode_mu_law(samples, config.n_quantize)
                sf.write(args.outdir + "/" + feat_id + ".wav", wav, args.fs, "PCM_16")
                logging.info("wrote %s.wav in %s." % (feat_id, args.outdir))

    # parallel decode
    if args.n_gpus > 0:
        logging.info("gpu decoding")
        processes = []
        gpu = 0
        for i, feat_list in enumerate(feat_lists):
            p = mp.Process(target=gpu_decode, args=(feat_list, gpu,))
            p.start()
            processes.append(p)
            gpu += 1
            if (i + 1) % args.n_gpus == 0:
                gpu = 0
    else:
        logging.info("cpu decoding")
        processes = []
        for i, feat_list in enumerate(feat_lists):
            p = mp.Process(target=cpu_decode, args=(feat_list,))
            p.start()
            processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
