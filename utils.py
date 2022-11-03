import torch
import torch.nn as nn
import numpy as np
import clip

class Embeder:
    
    def __init__(self, encoder):
        self.encoder = encoder
  
    def encode(self, x):
        pass
    
    
class ImageEmbeder(Embeder):

    def __init__(self, encoder, transforms=None):
        super().__init__(encoder)
        self.transforms = transforms

    def encode(self, images):
        if self.transforms is not None:
            images = [self.transforms(x) for x in images]
        image_input = torch.tensor(np.stack(images))
        with torch.no_grad():
            return self.encoder.encode_image(image_input).type(torch.float)
    
    
class TextEmbeder(Embeder):

    def __init__(self, encoder):
        super().__init__(encoder)

    def encode(self, texts):
        text_tokens = clip.tokenize([desc[:77] for desc in texts])
        with torch.no_grad():
            return self.encoder.encode_text(text_tokens).type(torch.float)
    
    
class SATextEmbeder(Embeder):

    def __init__(self, encoder, tokenizer):
        super().__init__(encoder)
        self.tokenizer = tokenizer

    def encode(self, texts):
        text_input = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=[desc[:77] for desc in texts],
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = 79,  # maximum length of a sentence
            padding='max_length',  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )
        with torch.no_grad():
            text_encoded = self.encoder(**text_input)
        return text_encoded.logits.type(torch.float)
    