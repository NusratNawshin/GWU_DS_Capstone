# IMPORTS
import math
import random
from collections import Counter

import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import spacy
import pandas as pd
from PIL import Image
from torchvision.models import ResNet18_Weights
spacy_eng = spacy.load("en_core_web_sm")
from tqdm import tqdm
import pickle
torch.manual_seed(17)
from train_and_validation import ImageCaptionModel,PositionalEncoding,FlickerDataSetResnet

### VARIABLES
# train_file_path = '../data/tokenized_annotation/train.pkl'
# train_image_path = "../data/images/train/"

val_file_path = '../data/tokenized_annotation/val.pkl'
val_image_path = "../data/images/val/"

vocabs = pd.read_pickle('model/vocabsize.pkl')
#
index_to_word=vocabs["index_to_word"]
word_to_index = vocabs["word_to_index"]
max_seq_len = vocabs["max_seq_len"]
vocab_size=vocabs["vocab_size"]


# max_seq_len = 46
IMAGE_SIZE = 224
# EPOCH = 60

###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

###
# TRAIN
# train = pd.read_csv(train_file_path)
# train = pd.read_pickle(train_file_path)
# print(f"Length of Train set: {len(train)}")
# print(train.tail(3))
# VAL
# valid = pd.read_csv(val_file_path)
valid = pd.read_pickle(val_file_path)
unique_valid = valid[['Name']].drop_duplicates()
# unique_valid=unique_valid[:15]
# print(valid.columns)
# for i in range(len(valid)):
#     print(valid[i])
print(f"Length of validation set: {len(valid)}")
# print(valid.head(3))

###############################################################
#                    Generate Caption
###############################################################

model = torch.load('model/BestModel')
start_token = word_to_index['<start>']
end_token = word_to_index['<end>']
pad_token = word_to_index['<pad>']
max_seq_len = 46
# print(start_token, end_token, pad_token)


valid_img_embed = pd.read_pickle('model/EncodedImageValidResNet.pkl')
# print(valid_img_embed)

def generate_caption(K, img_nm, img_loc):
    # img_loc = img_loc+str(img_nm)
    # image = Image.open(img_loc).convert("RGB")
    # plt.imshow(image)

    model.eval()
    # valid_img_df = valid[valid['Name']==img_nm]
    # print(valid.query('Name==@img_nm')["index"])
    # print(valid.index[valid["Name"]==img_nm].tolist())
    captionindex=valid.index[valid["Name"]==img_nm].tolist()
    # print(valid["Caption"][indexs[0]])
    # print(captionindex)
    actual_caption=[]
    for i in range(len(captionindex)):
        actual_caption.append(valid["Caption"][captionindex[i]])

    # print("Actual Caption : ")
    # print(valid_img_df)
    # actual_caption=valid_img_df['Caption'].to_string(index=False)
    # print(valid_img_df['Caption'])
    img_embed = valid_img_embed[img_nm].to(device)
    # print(img_embed)


    img_embed = img_embed.permute(0,2,3,1)
    img_embed = img_embed.view(img_embed.size(0), -1, img_embed.size(3))


    input_seq = [pad_token]*max_seq_len
    input_seq[0] = start_token

    input_seq = torch.tensor(input_seq).unsqueeze(0).to(device)
    predicted_sentence = []
    with torch.no_grad():
        # for eval_iter in range(0, max_seq_len):
        for eval_iter in range(0, max_seq_len-1):

            output, padding_mask = model.forward(img_embed, input_seq)

            output = output[eval_iter, 0, :]
            # print(output.tolist())
            # print(len(output.tolist()))


            values = torch.topk(output, K).values.tolist()
            indices = torch.topk(output, K).indices.tolist()
            # print("values")
            # print(values)
            # print("Indices")
            # print(indices)

            next_word_index = random.choices(indices, values, k = 1)[0]
            # print("next word index")
            # print(next_word_index)
            next_word = index_to_word[next_word_index]
            # print("next word")
            # print(next_word)

            input_seq[:, eval_iter+1] = next_word_index


            if next_word == '<end>' :
                break

            predicted_sentence.append(next_word)
    # print("\n")
    # print("Predicted caption : ")
    # print(" ".join(predicted_sentence+['.']))
    predicted_sentence = " ".join(predicted_sentence)
    # print(type(actual_caption))
    return [img_nm,actual_caption, predicted_sentence]

# predictions=[]
image_names=[]
actual_captions=[]
predicted_captions=[]
for i in range(len(unique_valid)):
# for i in range(0, 2):
    pred = generate_caption(1, unique_valid.iloc[i]['Name'], val_image_path)
    # print(unique_valid.iloc[i]['Name'])
    # print(pred)
    print("-"*50)
    print("Image Name: "+pred[0])
    print("Actual Caption: "+str(pred[1]))
    print("Predicted Caption: "+pred[2])
    print("-"*50)
    image_names.append(pred[0])
    actual_captions.append(pred[1])
    predicted_captions.append(pred[2])

# print(predictions[0])

pred_df = pd.DataFrame(
    {'Name': image_names,
     'actual_captions': actual_captions,
     'predicted_captions': predicted_captions,
     })
print(pred_df.dtypes)
# pred_df = pd.DataFrame(predictions)
# print(pred_df.shape)
pred_df.to_csv('results/results.csv', index=False)
