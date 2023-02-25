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

### VARIABLES
train_file_path = "../data/annotations/train.csv"
train_image_path = "../data/images/train/"

val_file_path = "../data/annotations/val.csv"
val_image_path = "../data/images/val/"


max_seq_len = 46
IMAGE_SIZE = 224
EPOCH = 30

###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

###
# TRAIN
train = pd.read_csv(train_file_path, sep=',', skipinitialspace = True)
print(f"Length of Train set: {len(train)}")
print(train.tail(3))
# VAL
valid = pd.read_csv(val_file_path)
print(f"Length of validation set: {len(valid)}")
print(valid.head(3))

#################################
# Concat train & validation dataframes

df = pd.concat([train,valid],ignore_index=True)
print(f"Length of total df: {len(df)}")
print(df[20559:20563])
print('-'*60)
print('-'*60)
#################################
# # CAPTION PREPROCESSING

# remove single character
df["cleaned_caption"] = df["Caption"].str.replace(r'\b[a-zA-Z] \b', '', regex=True)
# valid["cleaned_caption"] = valid["Caption"].str.replace(r'\b[a-zA-Z] \b', '', regex=True)

# lower characters
df['cleaned_caption'] = df['cleaned_caption'].apply(str.lower)
# valid['cleaned_caption'] = valid['cleaned_caption'].apply(str.lower)

print('-'*60)
print(df['Caption'][:4])
print(df['cleaned_caption'][:4])
print('-'*60)
# Get maximum length of caption sequence
df['cleaned_caption'] = df['cleaned_caption'].apply(lambda caption : ['<start>'] + [word.lower() if word.isalpha() else '' for word in caption.split(" ")] + ['<end>'])
df['seq_len'] = df['cleaned_caption'].apply(lambda x : len(x))
max_seq_len = df['seq_len'].max()
print(f"Maximum length of sequence: {max_seq_len}")
print('-'*60)
print(f"Data with caption sequence length more than 45")
print(df[df['seq_len'] > 45])
print('-'*60)
print(f"Tokenized caption:- ")
print(df['cleaned_caption'][0])
df.drop(['seq_len'], axis = 1, inplace = True)
print('-'*60)

df['seq_len'] = df['cleaned_caption'].apply(lambda x : len(x))
# Considering the max sequence length to be 46
for i in range(len(df['cleaned_caption'])):
    if "" in df['cleaned_caption'][i]:
        df.iloc[i]['cleaned_caption']=df['cleaned_caption'][i].remove("")

    if(len(df['cleaned_caption'][i])>46):
        temp =df['cleaned_caption'][i]
        temp = temp[:45]
        temp.append("<end>")
        df._set_value(i,'cleaned_caption',temp)

print(df['cleaned_caption'][0])
df['seq_len'] = df['cleaned_caption'].apply(lambda x : len(x))
max_seq_len = df['seq_len'].max()
print(f"Maximum length of sequence: {max_seq_len}")
df.drop(['seq_len'], axis = 1, inplace = True)
df['cleaned_caption'] = df['cleaned_caption'].apply(lambda caption : caption + ['<pad>']*(max_seq_len-len(caption)) )

# Create Vocabulary
word_list = df['cleaned_caption'].apply(lambda x : " ".join(x)).str.cat(sep = ' ').split(' ')
word_dict = Counter(word_list)
word_dict =  sorted(word_dict, key=word_dict.get, reverse=True)

print('-'*60)
print(f"Length of word dict: {len(word_dict)}")
vocab_size = len(word_dict)
print(f"Vocab Size: {vocab_size}")
print('-'*60)

# word to indexing
index_to_word = {index: word for index, word in enumerate(word_dict)}
word_to_index = {word: index for index, word in enumerate(word_dict)}
# print(len(index_to_word), len(word_to_index))

# Covert sequence of tokens to IDs
df['text_seq']  = df['cleaned_caption'].apply(lambda caption : [word_to_index[word] for word in caption] )
print(df.head(5))
print('-'*60)

# Train-Validation split
print('-'*60)
train = df[:20560]
valid = df[20560:]
valid.reset_index(inplace=True,drop=True)
print("Train DF: ")
print(train.tail(3))
print("Validation DF: ")
print(valid.head(3))
print(type(train), len(train))
print(type(valid), len(valid))
print('-'*60)

#############################################################################################################
print("--IMAGE PROCESSING--")
train_trnsform=transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize([IMAGE_SIZE,IMAGE_SIZE]),
    # transforms.RandomRotation(degrees=45),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomPerspective(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomInvert(0.5),
    # transforms.RandomAdjustSharpness(0.5),
    # transforms.ColorJitter(),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
])
val_trnsform=transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize([IMAGE_SIZE,IMAGE_SIZE]),
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
])

# class Vocabulary:
#     def __init__(self, freq_threshold):
#         self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
#         self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
#         self.freq_threshold = freq_threshold
#
#     def __len__(self):
#         return len(self.itos)
#
#     @staticmethod
#     def tokenizer_eng(text):
#         return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
#
#     def build_vocabulary(self, sentence_list):
#         frequencies = {}
#         idx = 4
#
#         for sentence in sentence_list:
#             for word in self.tokenizer_eng(sentence):
#                 if word not in frequencies:
#                     frequencies[word] = 1
#
#                 else:
#                     frequencies[word] += 1
#
#                 if frequencies[word] == self.freq_threshold:
#                     self.stoi[word] = idx
#                     self.itos[idx] = word
#                     idx += 1
#
#     def numericalize(self, text):
#         tokenized_text = self.tokenizer_eng(text)
#
#         return [
#             self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
#             for token in tokenized_text
#         ]
#


class CustomDataset(Dataset):
    def __init__(self, csv_file, image_path, transform=None, freq_threshold=5):
        self.image_path = image_path
        self.data = pd.read_csv(csv_file)
        self.data = self.data
        # self.captions = self.data['Caption']
        self.transform = transform


    def __len__(self):
        return len(self.data)

    # def get_batch_texts(self, index):
    #     # Fetch a batch of inputs
    #     return self.texts[index]

    def __getitem__(self, index):
        # IMAGES
        img_name = self.data.iloc[index, 0]
        img_path = str(self.image_path+img_name)

        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        # numericalized_caption = [self.vocab.stoi["<SOS>"]]
        # numericalized_caption += self.vocab.numericalize(caption)
        # numericalized_caption.append(self.vocab.stoi["<EOS>"])
        # print(torch.tensor(numericalized_caption))

        # return img, torch.tensor(numericalized_caption),img_name
        return img,img_name

# class MyCollate:
#     def __init__(self, pad_idx):
#         self.pad_idx = pad_idx
#
#     def __call__(self, batch):
#         imgs = [item[0].unsqueeze(0) for item in batch]
#         imgs = torch.cat(imgs, dim=0)
#         targets = [item[1] for item in batch]
#         print(targets)
#         targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
#         img_name = [item[2] for item in batch]
#         print("AFTER PADDING")
#         print(targets)
#
#         return imgs, targets, img_name

# Train dataset

train_image_dataset = CustomDataset(train_file_path, train_image_path, transform = train_trnsform)
# pad_idx = train_image_dataset.vocab.stoi["<PAD>"]
train_image_dataloader = DataLoader(train_image_dataset, batch_size=1, shuffle=False)
# train_image_dataloader = DataLoader(train_image_dataset, batch_size=1, shuffle=True, collate_fn=MyCollate(pad_idx=pad_idx))

# cnt=0
# for batch in train_dataloader:
#     print(f"batch {cnt}")
#     images, labels = batch
#     cnt +=1
    # Do something with the data

# for idx, (imgs, captions,image_name) in enumerate(train_dataloader):
#     print(image_name)
#     print(captions.shape)

# validation dataset

val_image_dataset = CustomDataset(val_file_path, val_image_path, transform = val_trnsform)
# pad_idx = val_image_dataset.vocab.stoi["<PAD>"]
val_image_dataloader = DataLoader(val_image_dataset, batch_size=1, shuffle=False)
# val_image_dataloader = DataLoader(val_image_dataset, batch_size=1, shuffle=True, collate_fn=MyCollate(pad_idx=pad_idx))

# for idx, (imgs, captions) in enumerate(test_dataloader):
#     print(imgs.shape)
#     print(captions.shape)




# RESNET
resnet18 = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
resnet18.eval()
print(list(resnet18._modules))

resNet18Layer4 = resnet18._modules.get('layer4').to(device)


def get_vector(t_img):
    t_img = Variable(t_img)
    my_embedding = torch.zeros(1, 512, 7, 7)

    def copy_data(m, i, o):
        my_embedding.copy_(o.data)

    h = resNet18Layer4.register_forward_hook(copy_data)
    resnet18(t_img)

    h.remove()
    return my_embedding


extract_imgFtr_ResNet_train = {}
print("Extracting features from Train set:")
for imgs,image_name in tqdm(train_image_dataloader):
    t_img = imgs.to(device)
    embdg = get_vector(t_img)
    extract_imgFtr_ResNet_train[image_name[0]] = embdg
# print(extract_imgFtr_ResNet_train)
# print(tokenized_caption_train)

a_file = open("model/EncodedImageTrainResNet.pkl", "wb")
pickle.dump(extract_imgFtr_ResNet_train, a_file)
a_file.close()



extract_imgFtr_ResNet_valid = {}
print("Extracting features from Validation set:")
for imgs,image_name in tqdm(val_image_dataloader):
    t_img = t_img.to(device)
    embdg = get_vector(t_img)

    extract_imgFtr_ResNet_valid[image_name[0]] = embdg

a_file = open("model/EncodedImageValidResNet.pkl", "wb")
pickle.dump(extract_imgFtr_ResNet_valid, a_file)
a_file.close()

#################################
class FlickerDataSetResnet():
    def __init__(self, data, pkl_file):
        self.data = data
        self.encodedImgs = pd.read_pickle(pkl_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        caption = self.data.iloc[idx]['text_seq']
        target_seq = caption[1:] + [0]

        image_name = self.data.iloc[idx]['Name']
        image_tensor = self.encodedImgs[image_name]
        image_tensor = image_tensor.permute(0, 2, 3, 1)
        image_tensor_view = image_tensor.view(image_tensor.size(0), -1, image_tensor.size(3))

        return torch.tensor(caption), torch.tensor(target_seq), image_tensor_view


train_dataset_resnet = FlickerDataSetResnet(train, 'model/EncodedImageTrainResNet.pkl')
train_dataloader_resnet = DataLoader(train_dataset_resnet, batch_size = 32, shuffle=True)


# Validation set preprocessing


valid_dataset_resnet = FlickerDataSetResnet(valid, 'model/EncodedImageValidResNet.pkl')
valid_dataloader_resnet = DataLoader(valid_dataset_resnet, batch_size = 32, shuffle=True)

# for idx, (imgs, captions,image_name) in enumerate(train_dataloader_resnet):
#     print(image_name)

# Positional Encoding
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.pe.size(0) < x.size(0):
            self.pe = self.pe.repeat(x.size(0), 1, 1).to(device)
        self.pe = self.pe[:x.size(0), :, :]

        x = x + self.pe
        return self.dropout(x)


# word_list = df['Caption'].apply(lambda x : " ".join(x)).str.cat(sep = ' ').split(' ')
# word_dict = Counter(word_list)
# word_dict =  sorted(word_dict, key=word_dict.get, reverse=True)
# print(f"Vocab Size: {len(word_dict)}")
# vocab_size = len(word_dict)

class ImageCaptionModel(nn.Module):
    def __init__(self, n_head, n_decoder_layer, vocab_size, embedding_size):
        super(ImageCaptionModel, self).__init__()
        self.pos_encoder = PositionalEncoding(embedding_size, 0.1)
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(d_model=embedding_size, nhead=n_head)
        self.TransformerDecoder = nn.TransformerDecoder(decoder_layer=self.TransformerDecoderLayer,
                                                        num_layers=n_decoder_layer)
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.last_linear_layer = nn.Linear(embedding_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.last_linear_layer.bias.data.zero_()
        self.last_linear_layer.weight.data.uniform_(-initrange, initrange)

    def generate_Mask(self, size, decoder_inp):
        decoder_input_mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        decoder_input_mask = decoder_input_mask.float().masked_fill(decoder_input_mask == 0, float('-inf')).masked_fill(
            decoder_input_mask == 1, float(0.0))

        decoder_input_pad_mask = decoder_inp.float().masked_fill(decoder_inp == 0, float(0.0)).masked_fill(
            decoder_inp > 0, float(1.0))
        decoder_input_pad_mask_bool = decoder_inp == 0

        return decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool

    def forward(self, encoded_image, decoder_inp):
        encoded_image = encoded_image.permute(1, 0, 2)

        decoder_inp_embed = self.embedding(decoder_inp) * math.sqrt(self.embedding_size)

        decoder_inp_embed = self.pos_encoder(decoder_inp_embed)
        decoder_inp_embed = decoder_inp_embed.permute(1, 0, 2)

        decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool = self.generate_Mask(
            decoder_inp.size(1), decoder_inp)
        decoder_input_mask = decoder_input_mask.to(device)
        decoder_input_pad_mask = decoder_input_pad_mask.to(device)
        decoder_input_pad_mask_bool = decoder_input_pad_mask_bool.to(device)

        decoder_output = self.TransformerDecoder(tgt=decoder_inp_embed, memory=encoded_image,
                                                 tgt_mask=decoder_input_mask,
                                                 tgt_key_padding_mask=decoder_input_pad_mask_bool)

        final_output = self.last_linear_layer(decoder_output)

        return final_output, decoder_input_pad_mask

# MODEL TRAIN

ictModel = ImageCaptionModel(16, 4, vocab_size, 512).to(device)
optimizer = torch.optim.Adam(ictModel.parameters(), lr = 0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.8, patience=2, verbose = True)
criterion = torch.nn.CrossEntropyLoss(reduction='none')
min_val_loss = float('Inf')

for epoch in tqdm(range(EPOCH)):
    total_epoch_train_loss = 0
    total_epoch_valid_loss = 0
    total_train_words = 0
    total_valid_words = 0
    ictModel.train()

    ### Train Loop
    for caption_seq, target_seq, image_embed in train_dataloader_resnet:
        optimizer.zero_grad()

        image_embed = image_embed.squeeze(1).to(device)
        caption_seq = caption_seq.to(device)
        target_seq = target_seq.to(device)

        output, padding_mask = ictModel.forward(image_embed, caption_seq)
        output = output.permute(1, 2, 0)

        loss = criterion(output, target_seq)

        loss_masked = torch.mul(loss, padding_mask)

        final_batch_loss = torch.sum(loss_masked) / torch.sum(padding_mask)

        final_batch_loss.backward()
        optimizer.step()
        total_epoch_train_loss += torch.sum(loss_masked).detach().item()
        total_train_words += torch.sum(padding_mask)

    total_epoch_train_loss = total_epoch_train_loss / total_train_words

    ### Eval Loop
    ictModel.eval()
    with torch.no_grad():
        for caption_seq, target_seq, image_embed in valid_dataloader_resnet:
            image_embed = image_embed.squeeze(1).to(device)
            caption_seq = caption_seq.to(device)
            target_seq = target_seq.to(device)

            output, padding_mask = ictModel.forward(image_embed, caption_seq)
            output = output.permute(1, 2, 0)

            loss = criterion(output, target_seq)

            loss_masked = torch.mul(loss, padding_mask)

            total_epoch_valid_loss += torch.sum(loss_masked).detach().item()
            total_valid_words += torch.sum(padding_mask)

    total_epoch_valid_loss = total_epoch_valid_loss / total_valid_words

    # print("Epoch -> ", epoch, " Training Loss -> ", total_epoch_train_loss.item(), "Eval Loss -> ",
    #       total_epoch_valid_loss.item())
    print(f"Epoch -> {epoch},  Training Loss -> {total_epoch_train_loss.item():.3f} Eval Loss -> {total_epoch_valid_loss.item():.3f}")

    if min_val_loss > total_epoch_valid_loss:
        print("Writing Model at epoch ", epoch)
        torch.save(ictModel, 'model/BestModel')
        min_val_loss = total_epoch_valid_loss

    scheduler.step(total_epoch_valid_loss.item())

###############################################################
#                    Generate Caption
###############################################################

model = torch.load('model/BestModel')
start_token = word_to_index['<start>']
end_token = word_to_index['<end>']
pad_token = word_to_index['<pad>']
max_seq_len = 46
print(start_token, end_token, pad_token)


valid_img_embed = pd.read_pickle('model/EncodedImageValidResNet.pkl')

def generate_caption(K, img_nm, img_loc):
    img_loc = img_loc+str(img_nm)
    image = Image.open(img_loc).convert("RGB")
    plt.imshow(image)

    model.eval()
    valid_img_df = valid[valid['Name']==img_nm]
    print("Actual Caption : ")
    print(valid_img_df['Caption'])
    img_embed = valid_img_embed[img_nm].to(device)


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

            values = torch.topk(output, K).values.tolist()
            indices = torch.topk(output, K).indices.tolist()

            next_word_index = random.choices(indices, values, k = 1)[0]

            next_word = index_to_word[next_word_index]

            input_seq[:, eval_iter+1] = next_word_index


            if next_word == '<end>' :
                break

            predicted_sentence.append(next_word)
    print("\n")
    print("Predicted caption : ")
    print(" ".join(predicted_sentence+['.']))

generate_caption(1, valid.iloc[50]['Name'], val_image_path)

# ### REFERENCES
#     # https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset_txt/loader_customtext.py