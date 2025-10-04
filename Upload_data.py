
# Now you can open it as usual
with open(r"D:\Devops\Repos\LLMs-from-scratch\book.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

#print("Total number of characters:", len(raw_text))
#print(raw_text[:99])

from os import pipe
import re
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
#print(preprocessed[:30])
#print(len(preprocessed))



all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)



vocab = {token:integer for integer, token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
  print(item)
  if i >= 50:
    break
  

class SimpleTokenizerV1:
   def __init__(self, vocab):
       self.str_to_int = vocab
       self.int_to_str = {i:s for s,i in vocab.items()}
   
   def encode(self, text):
       preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
       preprocessed = [
           item.strip() for item in preprocessed if item.strip()
       ]
       ids = [self.str_to_int[s] for s in preprocessed]
       return ids
        
   def decode(self, ids):
       text = " ".join([self.int_to_str[i] for i in ids])
       # Replace spaces before the specified punctuations
       text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
       return text
    



tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)


tokenizer.decode(ids)




all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer,token in enumerate(all_tokens)}


len(vocab.items())



for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)



class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
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
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    

tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(text)


tokenizer.encode(text)


tokenizer.decode(tokenizer.encode(text))


import importlib.metadata
import tiktoken

print("tiktoken version:", importlib.metadata.version("tiktoken"))


tokenizer = tiktoken.get_encoding("gpt2")


text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)


strings = tokenizer.decode(integers)

print(strings)


integers = tokenizer.encode("Akwirw ier")
print(integers)

strings = tokenizer.decode(integers)
print(strings)


import tiktoken

# Initialize the encodings for GPT-2, GPT-3, and GPT-4
encodings = {
    "gpt2": tiktoken.get_encoding("gpt2"),
    "gpt3": tiktoken.get_encoding("p50k_base"),  # Commonly associated with GPT-3 models
    "gpt4": tiktoken.get_encoding("cl100k_base")  # Used for GPT-4 and later versions
}

# Get the vocabulary size for each encoding
vocab_sizes = {model: encoding.n_vocab for model, encoding in encodings.items()}

# Print the vocabulary sizes
for model, size in vocab_sizes.items():
    print(f"The vocabulary size for {model.upper()} is: {size}")
