import torch
from espnet2.bin.asr_inference import Speech2Text
import soundfile
from espnet2.tasks.asr import ASRTask
from espnet2.torch_utils.device_funcs import to_device
from transformers import BertTokenizer, BertModel

import sys
import os
import glob
import deepdish as dd

def get_data_iterator(speech2text, data_path, batch_size):
    data_path_and_name_and_type = [(data_path, "speech", "sound")]
    loader = ASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype='float32',
        batch_size=batch_size,
        num_workers=1,
        preprocess_fn=ASRTask.build_preprocess_fn(speech2text.asr_train_args, False),
        collate_fn=ASRTask.build_collate_fn(speech2text.asr_train_args, False),
        allow_variable_data_keys=False,
        inference=True,
    )
    return loader

def init_pretrained_bert_model(bert_type="bert-base-uncased"):
    tokenizer = BertTokenizer.from_pretrained(bert_type)
    model = BertModel.from_pretrained(bert_type, return_dict=True)
    return tokenizer, model

def get_key_to_transcript(text_file_path):
    key_to_transcript_dict = dict()
    with open(text_file_path, "r") as f:
        for line in f:
            key, transcript = line.strip().split(None, 1)
            key_to_transcript_dict[key] = transcript
    return key_to_transcript_dict

def check_existing_keys_set(output_dir):
    return set([os.path.basename(f).split(".")[0] for f in glob.glob(output_dir + "/*.h5")])

def main():
    librispeech_asr_path = "/home/prac0003/2_Modules/espnet/egs2/librispeech/asr1"
    exp_dir = f"{librispeech_asr_path}/exp/asr_train_asr_conformer7_n_fft512_hop_length256_raw_en_bpe5000_sp"
    asr_config = f"{exp_dir}/config.yaml"
    asr_model_file = f"{exp_dir}/valid.acc.best.pth"
    device = "cuda"
    audio_data_path = f"{librispeech_asr_path}/data/train_clean_100/wav.scp"
    text_data_path = f"{librispeech_asr_path}/data/train_clean_100/text"
    
    output_dir = "/scratch/prac0003/secondpass_exp/data"
    
    batch_size = 16
    
    speech2text = Speech2Text(
        asr_config,
        asr_model_file,
        lm_weight=0,
        nbest=10,
    )
    
    bert_tokenizer, bert_model = init_pretrained_bert_model()
    key_to_transcript_dict = get_key_to_transcript(text_data_path)
    existing_keys = check_existing_keys_set(output_dir)
    loader = get_data_iterator(speech2text, audio_data_path, batch_size)
    print() 
    for keys, batch in loader:
        #print("keys", keys)
        #print("batch", batch)
        #batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}
        
        batch = to_device(batch, device=device)
        results = speech2text.batch_inference(**batch)
        #assert(len(results) == batch_size)

        for i, r in enumerate(results):
            key = keys[i]
            if key in existing_keys:
                continue
            result, encoded_audio, speech_lengths = r
            #print("encoded", encoded_audio.size())
            #print("speech lengths", speech_lengths)
            hypotheses = []
            for h in result:
                hypotheses.append(h[0])
            
            #print("hypotheses", hypotheses)
            encoded_hypo = bert_tokenizer(hypotheses, padding=True, truncation=True, return_tensors='pt')
            bert_out = bert_model(**encoded_hypo)
            print("key", key)
            #print("encoded hypo", encoded_hypo)
            #print("bert_out last hidden state size", bert_out.last_hidden_state.size())
            #print("bert out pooler output", bert_out.pooler_output.size())
            
            # use attn mask to get the word length
            lm_lengths = torch.flatten(torch.sum(encoded_hypo.attention_mask, dim=1))
            
            # use key to get the original transcript
            transcript = key_to_transcript_dict[key]
            speech_length = speech_lengths[i].item()
            encoded_audio = encoded_audio[:speech_length, :]
            #print("encoded_audio size", encoded_audio.size())
            # save each into its own dict (?)
            save_dict = {
                'transcript': transcript,
                'hypotheses': hypotheses,
                'encoded_audio': encoded_audio[:speech_length, :],
                'speech_length': speech_length,
                'encoded_hypotheses': bert_out,
                'hypotheses_lengths': lm_lengths
            }
            #print(save_dict)
            dd.io.save(output_dir + "/" + key + ".h5", save_dict, compression=('blosc', 5))
            print("saved:", key)

if __name__=="__main__":
    main()
