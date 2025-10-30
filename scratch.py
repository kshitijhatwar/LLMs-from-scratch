############## Importing boook
##############
from google.colab import files
uploaded = files.upload()

with open("book.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
  
print("Total number of character:", len(raw_text))
print(raw_text[:99])

############################## Creating Tokenizer
##############################

import re
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)      ###saperating all the words
preprocessed = [item.strip() for item in preprocessed if item.strip()]    ### removing all the white spaces
print(preprocessed[:30])
print(len(preprocessed))

##### creating list of all unique tokens and sorting them alphabetacally, also adding endoftext and unki(to use for token not present in dictonary)
all_words = sorted(set(preprocessed))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)
print(vocab_size)

###Creating a dictonary mapping each integer with token IDs
vocab = {token:integer for integer,token in enumerate(all_words)}


#########

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)                      
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text









