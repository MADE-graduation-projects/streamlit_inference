import io
import os
from time import time

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import clip
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import easyocr

from utils import ImageEmbeder, TextEmbeder, SATextEmbeder

class NNModel(nn.Module):
    
    def __init__(self, image_input_dim, text_input_dim, sa_input_dim, hidden_layer_size, dropout, num_classes):
        super().__init__()
        
        self.sa_linear = nn.Linear(sa_input_dim, sa_input_dim)
        self.sa_activation = nn.ReLU()

        self.linear_1 = nn.Linear(image_input_dim + text_input_dim + sa_input_dim, hidden_layer_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.batchnorm_1 = nn.BatchNorm1d(hidden_layer_size)
        self.activation_1 = nn.ReLU()

        self.linear_2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.dropout_2 = nn.Dropout(dropout)
        self.batchnorm_2 = nn.BatchNorm1d(hidden_layer_size)
        self.activation_2 = nn.ReLU()

        self.last_linear = nn.Linear(hidden_layer_size, num_classes)
        
    def forward(self, x_image, x_text, x_sa):
        x_sa = self.sa_activation(self.sa_linear(x_sa))
        x = torch.cat([x_image, x_text, x_sa], dim=1)

        x = self.dropout_1(self.activation_1(self.batchnorm_1(self.linear_1(x))))

        x = self.dropout_2(self.activation_2(self.batchnorm_2(self.linear_2(x))))

        x = self.last_linear(x)

        return x
    
# Loading models
clip_model, clip_preprocess = clip.load("ViT-L/14", device="cpu")
sa_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sa_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment",
                                                           output_hidden_states=True)
model = NNModel(768, 768, 3, hidden_layer_size=256, dropout=0.66, num_classes=2)
model.load_state_dict(torch.load('clip_image_clip_text_sa_features_classifier_9_0.678.ckpt'))
model.eval()
# Initializing embeders
image_embeder = ImageEmbeder(clip_model, clip_preprocess)
text_embeder = TextEmbeder(clip_model)
sa_embeder = SATextEmbeder(sa_model, sa_tokenizer)

reader = easyocr.Reader(['en']) 

IMAGES_PATH = "images/"

def get_random_image():
    images = os.listdir(IMAGES_PATH)
    n = len(images)
    ind = np.random.randint(low=0, high=n)
    return Image.open(IMAGES_PATH + images[ind])

def get_text_from_image(image):
    result = reader.readtext(np.array(image))
    texts = [item[-2] for item in result]
    return ' '.join(texts)

def inference(img):
    start = time()
    text = get_text_from_image(img)
    image_embeding = image_embeder.encode([img])
    text_embeding = text_embeder.encode([text])
    sa_embeding = sa_embeder.encode([text])
    with torch.no_grad():
        output = model(image_embeding, text_embeding, sa_embeding)
    result = int(output.argmax())
    finish = time()
    print("Inference took {} seconds".format(finish - start))
    return "This meme is considered as toxic" if result else "This meme is considered as NOT toxic"



# uploaded_image_file = st.file_uploader("Upload a meme image")
# image = None
# if uploaded_image_file is not None:
#     image_bytes_data = uploaded_image_file.getvalue()
#     image = Image.open(io.BytesIO(image_bytes_data))

if st.button('Get random image and predict if it is toxic'):
    image = get_random_image()
    st.image(image, caption=inference(image))
            