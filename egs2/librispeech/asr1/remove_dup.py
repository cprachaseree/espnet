# remove duplicates if same sentence from train set to new 
import os
import shutil

path = "/home/prac0003/2_Modules/espnet/egs2/librispeech/asr1/data"
test_file = os.path.join(path, "test_2", "text")
train_file = os.path.join(path, "train_2", "text")
out_file = os.path.join(path, "train_3", "text")

print("copying")
shutil.copytree(os.path.join(path, "train_2"), os.path.join(path, "train_3"))
print("done copying")

print("creating test")
sentences = set()
with open(test_file, "r") as f:
    for line in f:
        sent = line.split(" ", 1)[1]
        sentences.add(sent)

print("done creating")
print("creating new train")
with open(train_file, "r") as tf:
    with open(out_file, "w") as of:
        for train_line in tf:
            _, sent = train_line.split(" ", 1)
            if sent not in sentences:
                of.write(train_line)
            sentences.add(sent)
print("done new train")
