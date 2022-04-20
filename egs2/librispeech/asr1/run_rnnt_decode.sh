#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960"
valid_set="dev"
#test_sets="test_clean test_other dev_clean dev_other"
test_sets="test"

asr_config=conf/train_rnnt_conformer.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_rnnt_conformer.yaml

./asr.sh \
    --lang en \
    --ngpu 0 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" \
    --bpe_train_text "data/${train_set}/text" "$@" \
    --stage 12 --stop_stage 16 --nj 8 --inference_nj 64 \
    --num_splits_asr 16 --num_splits_lm 1 --use-lm false \
    --inference_asr_model valid.loss.ave.pth \
    --lm_tag train_lm_en_bpe5000_tedlium \
    --audio_format flac                                 \
    --feats_type raw    
