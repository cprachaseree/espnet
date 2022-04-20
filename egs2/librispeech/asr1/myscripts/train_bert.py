from transformers import BertTokenizer, BertLMHeadModel
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class PseudoPPLDataset(Dataset):
    def __init__(self, data_path, tokenizer, model):
        self.data_path = data_path
        self._len = 0
        self.tokenizer = tokenizer
        self.model = model
    
    def __len__(self):
        return self._len
    
    def __get__item(self, idx):
        return 0
    
    def create_dataset(self):
        with open(self.data_path, "r") as f:
            for line in f:
                print("line", line)
                '''
                line_split = line.split()[1:]
                batch = []
                for i in range(len(line_split)):
                    new_line = line_split.copy()
                    new_line[i] = self.tokenizer.mask_token
                    batch.append(" ".join(new_line))

                encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                encoded_original = self.tokenizer(" ".join(line_split))
                print("word ids", encoded_original["input_ids"])
                batch_original = torch.tensor(encoded_original["input_ids"]).unsqueeze(0)
                print("len(batch)", len(batch))
                batch_original = batch_original.repeat(len(batch), 1)
                print("batch_original", batch_original)            
    
                output = self.model(**encoded_input, labels=batch_original)
                print("output.loss", output.loss)
                print("output.logits.size()", output.logits.size())
                mask_token_index = torch.where(encoded_input["input_ids"] == tokenizer.mask_token_id)[1]
                print("mask_token_index", mask_token_index)
                #selected = torch.index_select(output.logits, 2, torch.tensor(encoded_original["input_ids"]))
                #print("selected", selected)
                '''
                line = " ".join(line.split()[1:])
                encoded_line = self.tokenizer(line, return_tensors='pt')
                #print("encoded line", encoded_line)
                batch_labels = []
                batch_inputs = []
                line_ids = encoded_line['input_ids'][0]
                #print(line_ids)
                line_ids_len = len(line_ids)
                for i in range(line_ids_len - 2):
                    label = [-100] * line_ids_len
                    inputs = line_ids.clone() 
                    label[i+1] = inputs[i+1]
                    inputs[i+1] = self.tokenizer.mask_token_id
                    batch_labels.append(label)
                    batch_inputs.append(inputs)
                batch_labels = torch.LongTensor(batch_labels)
                #print(batch_labels)
                #print(batch_inputs)
                batch_size = len(batch_labels)
                batch_encoded_input = {
                    'input_ids': torch.stack(batch_inputs),
                    'token_type_ids': encoded_line['token_type_ids'].repeat(batch_size, 1),
                    'attention_mask': encoded_line['attention_mask'].repeat(batch_size, 1)
                }
                #print(batch_encoded_input)
                        
                output = self.model(**batch_encoded_input, labels=batch_labels)
                print(output.logits.size())
                print(output.loss)
                softmaxed_logits = torch.nn.functional.softmax(output.logits, dim=2)
                selected = torch.index_select()
                    
                    
                

def init_pretrained_model(bert_type="bert-base-uncased"):
    tokenizer = BertTokenizer.from_pretrained(bert_type)
    model = BertLMHeadModel.from_pretrained(bert_type)
    return tokenizer, model

def get_pseudoppl():
    pass

def train_cls_ppl():
	# https://huggingface.co/docs/transformers/training
	pass

if __name__ == "__main__":
    tokenizer, model = init_pretrained_model()
    d = PseudoPPLDataset(data_path="/home/prac0003/2_Modules/espnet/egs2/librispeech/asr1/myscripts/test.txt", tokenizer=tokenizer, model=model)
    d.create_dataset()
