import espnet2
import torch
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.torch_utils.load_pretrained_model import load_pretrained_model
from espnet2.tasks.asr import ASRTask
import yaml
from argparse import Namespace
from espnet2.torch_utils.device_funcs import to_device

def init_model(config_file, pretrained_file, device):
    print("initializing model")
    asr_model, asr_train_args = ASRTask.build_model_from_file(
        config_file, pretrained_file, device
    )
    print(asr_train_args)
    print(asr_model)
    asr_model.eval()
    return asr_model, asr_train_args

def get_data_iterator(data_path):
    data_path_and_name_and_type = [(data_path, "speech", "sound")]
    loader = ASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype='float32',
        batch_size=32,
        num_workers=1,
        preprocess_fn=ASRTask.build_preprocess_fn(asr_train_args, False),
        collate_fn=ASRTask.build_collate_fn(asr_train_args, False),
        allow_variable_data_keys=False,
        inference=True,
    )
    return loader

def train(loader, asr_model):
    # build data iterator
    loader = get_data_iterator(data_path)
    for keys, batch in loader:
        print("keys", keys)
        print("batch", batch)
        batch = to_device(batch, device=device)
        encoded_speech, speech_lengths = asr_model.encode(**batch)
        print("encoded", encoded)
        print("encoded_speech.size()", encoded_speech)
        print("speech_lengths.size()", speech_lengths)
        break

if __name__ == "__main__":
    exp_dir = "/home/prac0003/2_Modules/espnet/egs2/librispeech/asr1/exp/asr_train_asr_conformer7_n_fft512_hop_length256_raw_en_bpe5000_sp"
    config_file = f"{exp_dir}/config.yaml"
    pretrained_file = f"{exp_dir}/valid.acc.best.pth"
    device = "cuda"
    data_path = "/home/prac0003/2_Modules/espnet/egs2/librispeech/asr1/data/train_960/wav.scp"
    
    # init model based on pretrained and config
    asr_model, asr_train_args = init_model(config_file, pretrained_file, device)
    train(loader, asr_model)
