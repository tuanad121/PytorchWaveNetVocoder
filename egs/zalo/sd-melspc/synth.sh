#!/bin/bash
############################################################
#           SCRIPT TO BUILD SD WAVENET VOCODER             #
############################################################

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# USER SETTINGS {{{
#######################################
#           STAGE SETTING             #
#######################################
# {{{
# 0: data preparation step
# 1: feature extraction step
# 2: statistics calculation step
# 3: noise shaping step
# 4: training step
# 5: decoding step
# 6: restoring noise shaping step
# }}}
stage=0123456

#######################################
#          FEATURE SETTING            #
#######################################
# {{{
# shiftms: shift length in msec (default=5)
# fftl: fft length (default=1024)
# highpass_cutoff: highpass filter cutoff frequency (if 0, will not apply)
# mspc_dim: dimension of mel-spectrogram
# mcep_dim: dimension of mel-cepstrum (only used for noise shaping)
# mcep_alpha: all pass filter constant (only used for noise shaping)
# mag: coefficient of noise shaping (default=0.5)
# n_jobs: number of parallel jobs
# }}}
feature_type=melspc
shiftms=5
fftl=1024
highpass_cutoff=70
fs=16000
mspc_dim=80
mcep_dim=25
mcep_alpha=0.42
mag=0.5
n_jobs=20

#######################################
#          TRAINING SETTING           #
#######################################
# {{{
# n_gpus: number of gpus
# spk: target spekaer in arctic
# n_quantize: number of quantization
# n_aux: number of aux features
# n_resch: number of residual channels
# n_skipch: number of skip channels
# dilation_depth: dilation depth (e.g. if set 10, max dilation = 2^(10-1))
# dilation_repeat: number of dilation repeats
# kernel_size: kernel size of dilated convolution
# lr: learning rate
# weight_decay: weight decay coef
# iters: number of iterations
# batch_length: batch length
# batch_size: batch size
# checkpoints: save model per this number
# use_upsampling: true or false
# use_noise_shaping: true or false
# use_speaker_code: true or false
# resume: checkpoint to resume
# }}}
n_gpus=1
spk=f1
n_quantize=256
n_aux=80
n_resch=512
n_skipch=256
dilation_depth=10
dilation_repeat=3
kernel_size=2
lr=1e-4
weight_decay=0.0
iters=200000
batch_length=20000
batch_size=1
checkpoints=10000
use_upsampling=true
use_noise_shaping=true
resume=

#######################################
#          DECODING SETTING           #
#######################################
# {{{
# outdir: directory to save decoded wav dir (if not set, will automatically set)
# checkpoint: full path of model to be used to decode (if not set, final model will be used)
# config: model configuration file (if not set, will automatically set)
# feats: list or directory of feature files
# n_gpus: number of gpus to decode
# }}}
outdir=
checkpoint=
config=
feats=
decode_batch_size=32

#######################################
#            OHTER SETTING            #
#######################################
ARCTIC_DB_ROOT=downloads
tag=

# parse options
. parse_options.sh

# set params
train=tr_${spk}
eval=ev_${spk}

# stop when error occured
set -e
# }}}

# STAGE 4 {{{
# set variables
if [ ! -n "${tag}" ];then
    expdir=exp/tr_arctic_16k_sd_melspc_${spk}_nq${n_quantize}_na${n_aux}_nrc${n_resch}_nsc${n_skipch}_ks${kernel_size}_dp${dilation_depth}_dr${dilation_repeat}_lr${lr}_wd${weight_decay}_bl${batch_length}_bs${batch_size}
    if ${use_noise_shaping};then
        expdir=${expdir}_ns
    fi
    if ${use_upsampling};then
        expdir=${expdir}_up
    fi
else
    expdir=exp/tr_arctic_${tag}
fi

# }}}


# STAGE 5 {{{
if echo ${stage} | grep -q 5; then
    echo "###########################################################"
    echo "#               WAVENET DECODING STEP                     #"
    echo "###########################################################"
    [ ! -n "${outdir}" ] && outdir=${expdir}/wav
    [ ! -n "${checkpoint}" ] && checkpoint=${expdir}/checkpoint-final.pkl
    [ ! -n "${config}" ] && config=${expdir}/model.conf
    [ ! -n "${feats}" ] && feats=data/${eval}/feats.scp
    ${cuda_cmd} --gpu ${n_gpus} "${outdir}/log/decode.log" \
        decode.py \
            --n_gpus ${n_gpus} \
            --feats ${feats} \
            --stats data/${train}/stats.h5 \
            --outdir "${outdir}" \
            --checkpoint "${checkpoint}" \
            --config "${config}" \
            --fs ${fs} \
            --batch_size ${decode_batch_size}
fi
# }}}



# STAGE 6 {{{
if echo ${stage} | grep -q 6 && ${use_noise_shaping}; then
    echo "###########################################################"
    echo "#             RESTORE NOISE SHAPING STEP                  #"
    echo "###########################################################"
    [ ! -n "${outdir}" ] && outdir=${expdir}/wav
    find "${outdir}" -name "*.wav" | sort > data/${eval}/wav_generated.scp
    ${train_cmd} --num-threads ${n_jobs} exp/noise_shaping/noise_shaping_restore_mcep_${eval}.log \
        noise_shaping.py \
            --waveforms data/${eval}/wav_generated.scp \
            --stats data/${train}/stats.h5 \
            --writedir "${outdir}_restored" \
            --feature_type mcep \
            --fs ${fs} \
            --shiftms ${shiftms} \
            --fftl ${fftl} \
            --mcep_alpha ${mcep_alpha} \
            --mag ${mag} \
            --n_jobs ${n_jobs} \
            --inv false
fi
# }}}
