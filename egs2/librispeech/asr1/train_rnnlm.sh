#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#train_set="train_960"
#valid_set="dev"
#test_sets="test_clean test_other dev_clean dev_other"

train_set="train_2"
valid_set="validate"
test_sets="test_2"

asr_config=conf/tuning/train_asr_conformer7_n_fft512_hop_length256.yaml
lm_config=conf/tuning/train_lm_adam_finetune.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang en \
    --ngpu 4 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "dump/raw/sgh_med_keyed_upper.txt" \
    --bpe_train_text "data/${train_set}/text" "$@" \
    --stage 6 --stop_stage 9 --nj 8 --inference_nj 32 \
    --num_splits_asr 16 --num_splits_lm 1 --use-lm true \
    --lm_tag finetune_sghnoaugmed \
    --lm_dev_text /home/prac0003/2_Modules/espnet/egs2/librispeech/asr1/dump/raw_medical/validate/text
