import os
import glob
import subprocess

original_file = "/home/prac0003/2_Modules/espnet/egs2/librispeech/asr1/data/train_clean_100/text"
output_keys = "/scratch/prac0003/secondpass_exp/data/"

# get keys into set
real_keys = set()
with open(original_file, "r") as f:
    for line in f:
        real_keys.add(line.split()[0])

avail_keys = set([os.path.basename(f).split(".")[0] for f in glob.glob(output_keys + "*")])
diff = avail_keys.difference(real_keys)
print(len(real_keys))
for k in real_keys:
    print(k)
    break

print(len(avail_keys))
print("diff", len(diff))

# remove diff
for d in diff:
    subprocess.run(["rm", f"{output_keys}{d}.h5"])

remain = len(set([os.path.basename(f).split(".")[0] for f in glob.glob(output_keys + "*")]))
print("remain", remain)
