# IMPORTS
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
# from torchviz import make_dot
from tqdm import tqdm
torch.manual_seed(17)

### VARIABLES
# train_file_path = '../data/tokenized_annotation/train.pkl'
train_file_path = '../data/BERT_tokenized_annotation/train.pkl'
# train_file_path = '../data/cleaned_BERT_tokenized_annotation/train.pkl'
train_image_path = "../data/images/train/"

# val_file_path = '../data/tokenized_annotation/val.pkl'
val_file_path = '../data/BERT_tokenized_annotation/val.pkl'
# val_file_path = '../data/cleaned_BERT_tokenized_annotation/val.pkl'
val_image_path = "../data/images/val/"

vocabs = pd.read_pickle('model/vocabsize.pkl')

index_to_word=vocabs["index_to_word"]
word_to_index = vocabs["word_to_index"]
max_seq_len = vocabs["max_seq_len"]
vocab_size=vocabs["vocab_size"]

EPOCH = 25

###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#################################
class DatasetLoader():
    def __init__(self, data, pkl_file):
        self.data = data
        self.encodedImgs = pd.read_pickle(pkl_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        caption = self.data.iloc[idx]['text_seq']
        target_seq = caption[1:] + [0]
        # target_seq = caption
        image_name = self.data.iloc[idx]['Name']
        # print(image_name)
        # print(caption)
        image_tensor = self.encodedImgs[image_name]
        image_tensor = image_tensor.permute(0, 2, 3, 1)
        image_tensor_view = image_tensor.view(image_tensor.size(0), -1, image_tensor.size(3))

        return torch.tensor(caption), torch.tensor(target_seq), image_tensor_view


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

class ImageCaptionModel(nn.Module):
    def __init__(self, n_head, n_decoder_layer, vocab_size, embedding_size):
        super(ImageCaptionModel, self).__init__()
        self.pos_encoder = PositionalEncoding(embedding_size, 0.1)
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(d_model=embedding_size, nhead=n_head, activation= 'relu')
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
        # print("encoder shape")
        # print(encoded_image.shape)

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


def train():
    ###
    # TRAIN
    # train = pd.read_csv(train_file_path)
    train = pd.read_pickle(train_file_path)
    # train = train[:32]
    print(f"Length of Train set: {len(train)}")
    print(train.tail(3))
    # VAL
    # valid = pd.read_csv(val_file_path)
    valid = pd.read_pickle(val_file_path)
    # valid=valid[:32]
    print(f"Length of validation set: {len(valid)}")
    print(valid.head(3))

    train_dataset_resnet = DatasetLoader(train, 'model/EncodedImageTrain.pkl')
    train_dataloader_resnet = DataLoader(train_dataset_resnet, batch_size=32, shuffle=True)

    valid_dataset_resnet = DatasetLoader(valid, 'model/EncodedImageValid.pkl')
    valid_dataloader_resnet = DataLoader(valid_dataset_resnet, batch_size=32, shuffle=True)

    # MODEL TRAIN
    ictModel = ImageCaptionModel(16, 8, vocab_size, 512).to(device)

    # End
    # ictModel = torch.load('model/BestModel')
    optimizer = torch.optim.Adam(ictModel.parameters(), lr=0.00001)
    # # Controling model visualization
    # break_point=0
    # optimizer = torch.optim.Adamax(ictModel.parameters(), lr=0.00001)
    # optimizer = torch.optim.SGD(ictModel.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=2, verbose=True)
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
            # print(caption_seq)
            target_seq = target_seq.to(device)
            # print(target_seq)

            output, padding_mask = ictModel.forward(image_embed, caption_seq)
            output = output.permute(1, 2, 0)
            # if(break_point==0):
            #     dot = make_dot((output, padding_mask), params=dict(ictModel.named_parameters()), show_attrs=True, show_saved=True)
            #     dot.render(filename="modelArchitecture/ictModel_visualization", format='svg')
            #
            #     print('saved')
            #     break_point=break_point+1

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

        print(
            f"Epoch -> {epoch},  Training Loss -> {total_epoch_train_loss.item():.3f} Eval Loss -> {total_epoch_valid_loss.item():.3f}")

        if min_val_loss > total_epoch_valid_loss:
            print("Writing Model at epoch ", epoch)
            torch.save(ictModel, 'model/BestModel')
            min_val_loss = total_epoch_valid_loss

        scheduler.step(total_epoch_valid_loss.item())



if __name__ == "__main__":
    train()


# ### REFERENCES
#     # https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset_txt/loader_customtext.py
