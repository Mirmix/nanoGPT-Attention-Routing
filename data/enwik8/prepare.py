"""
Prepare the enwik8 dataset for character-level language modeling.
Downloads the first 100M characters of Wikipedia and splits into:
- train.bin: first 90M chars
- val.bin: next 5M chars  
- test.bin: final 5M chars
"""
import os
import pickle
import requests
import numpy as np

# download the enwik8 dataset (first 100M characters of Wikipedia)
input_file_path = os.path.join(os.path.dirname(__file__), 'enwik8')
if not os.path.exists(input_file_path):
    print("Downloading enwik8 dataset...")
    data_url = 'http://mattmahoney.net/dc/enwik8.zip'
    import zipfile
    import tempfile
    
    # Download and extract
    response = requests.get(data_url)
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name
    
    with zipfile.ZipFile(tmp_file_path, 'r') as zip_ref:
        # List contents to see what's in the archive
        print("Archive contents:", zip_ref.namelist())
        # Extract all files
        zip_ref.extractall(os.path.dirname(__file__))
    
    # Clean up
    os.unlink(tmp_file_path)
    print("Download complete!")

# Read the first 100M characters
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()[:100_000_000]  # First 100M characters

print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train, validation, and test splits
n = len(data)
train_data = data[:90_000_000]  # First 90M chars
val_data = data[90_000_000:95_000_000]  # Next 5M chars
test_data = data[95_000_000:100_000_000]  # Final 5M chars

# encode all to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"test has {len(test_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
test_ids.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Dataset preparation complete!") 