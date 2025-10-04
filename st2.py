class SimpleTokenizerV1:
  def __init__(self, voclab):
    self.str_to_int = voclab
    self.int_to_str =  {i:s for s,i in voclab.items()}

  def encode(self, text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

    preprocessed = [
        item.strip() for item in preprocessed if item.strip()
    ]
    ids = [self.str_to_int[s] for s in preprocessed]
    return ids

  def decode(self, ids):
    text = " ".join([self.int_to_str[i] for i in ids])
    text = re.sub(r'([,.:;?_!"()\']|--|\s)', r'\1', text)
    return text
  


tokanizer = SimpleTokenizerV1(voclab)

text = """It's the last he painted, you know,"
          Mr. Gisburn said pride."""
ids = tokanizer.encode(text)
print(ids)
  
