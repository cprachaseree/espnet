import os
import glob
import deepdish as dd
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.keys = [os.path.basename(f).split(".")[0] for f in glob.glob(data_path+ "*")]
        self.key2data = {}

    def __getitem__(self, idx):
        key = self.keys[idx]
        if key in self.key2data.keys():
            data = self.key2data[key]
        else:
            file_path = self.data_path + key + ".h5"
            data = dd.io.load(file_path)
            self.key2data[key] = data
        return data

    def __len__(self):
        return len(self.keys)

def my_collate_fn(data):
    batch_encoded_audio = []
    batch_encoded_hypotheses = []
    batch_hypotheses = []
    batch_hypotheses_lengths = []
    batch_audio_lengths = []
    batch_transcript = []
    for d in data:
        encoded_audio = d['encoded_audio']
        print("encoded audio", encoded_audio.size()) 
        batch_encoded_audio.append(encoded_audio)
        encoded_hypotheses = d['encoded_hypotheses']
        encoded_hypotheses_hidden_state = encoded_hypotheses['last_hidden_state']
        encoded_hypotheses_pooler_out = encoded_hypotheses['pooler_output']
        encoded_hypotheses_concat = torch.cat(torch.unsqueeze(encoded_hypotheses_pooler_out, dim=1), encoded_hypotheses_hidden_state)
        batch_encoded_hypotheses.append(encoded_hypotheses_concat)
        print("text_last_hidden_state", encoded_hypotheses_hidden_state.size())
        print("text_pooler", encoded_hypotheses_pooler_out.size())
        hypotheses = d['hypotheses']
        print("hypotheses", hypotheses)
        batch_hypotheses.append(hypotheses)
        hypotheses_lengths = d['hypotheses_lengths']
        print("h lengths", hypotheses_lengths)
        batch_hypotheses_lengths.append(torch.add(hypotheses_lengths, 1))
        audio_length = d['speech_length']
        print("audio_length", audio_length)
        batch_audio_lengths.append(audio_length)
        transcript = d['transcript']
        print("transcript", transcript)
        batch_transcript.append(transcript)
    data = {
        "batch_encoded_audio": 
    }
    batch_encoded_audio = []
    batch_encoded_hypotheses = []
    batch_hypotheses = []
    batch_hypotheses_lengths = []
    batch_audio_lengths = []
    batch_transcript = []
    return data

def get_my_dataloader(data_path, batch_size, shuffle=True, collate_fn=None):
    dataset = MyDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

if __name__=="__main__":
    path = "/scratch/prac0003/secondpass_exp/data/"
    dataset = MyDataset(path)
    for d in dataset:
        print(d)
        break
    
