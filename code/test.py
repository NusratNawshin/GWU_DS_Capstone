# IMPORTS
import math
import os
import random
from collections import Counter
import wordninja
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
# from torchvision.models import ResNet50_Weights
# from torchvision.models import ResNet18_Weights
from torchvision.models import VGG16_Weights
spacy_eng = spacy.load("en_core_web_sm")
from tqdm import tqdm
import pickle
torch.manual_seed(17)
from train_and_validation import ImageCaptionModel,PositionalEncoding,DatasetLoader

### VARIABLES
test_file_path = '../data/annotations/test.csv'
test_image_path = "../data/images/val/"

# max_seq_len = 46
IMAGE_SIZE = 224
extract_feature = True
# EPOCH = 60

###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

###

test = pd.read_csv(test_file_path)
# test=test[:32]
unique_test = test[['Name']].drop_duplicates()


print(f"Length of test set: {len(test)}")

vocabs = pd.read_pickle('model/vocabsize.pkl')
#
index_to_word=vocabs["index_to_word"]
word_to_index = vocabs["word_to_index"]
max_seq_len = vocabs["max_seq_len"]
vocab_size=vocabs["vocab_size"]

###############################################################
#                    Extract Features
###############################################################

if os.path.exists('model/EncodedImageTestResNet.pkl'):
    pass
if extract_feature == False:
    pass
else:
    print("--IMAGE PROCESSING--")
    test_trnsform=transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize([IMAGE_SIZE,IMAGE_SIZE]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class CustomDataset(Dataset):
        def __init__(self, data, image_path, transform=None):
            self.image_path = image_path
            self.data = data
            # self.data = self.data[['Name']].drop_duplicates()
            # self.data = self.data[:32]
            self.transform = transform


        def __len__(self):
            return len(self.data)


        def __getitem__(self, index):
            # IMAGES
            img_name = self.data.iloc[index, 0]
            img_path = str(self.image_path+img_name)
            img = Image.open(img_path)
            if self.transform is not None:
                img = self.transform(img)
            return img,img_name

    # test dataset
    test_file = pd.read_csv(test_file_path)
    test_file = test_file[['Name']].drop_duplicates()
    test_image_dataset = CustomDataset(test_file, test_image_path, transform = test_trnsform)
    test_image_dataloader = DataLoader(test_image_dataset, batch_size=1, shuffle=False)

    # RESNET 18
    # resnet18 = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    # resnet18.eval()
    # print(list(resnet18._modules))
    # resNet18Layer4 = resnet18._modules.get('layer4').to(device)

    # RESNET 50
    # resnet50 = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    # resnet50.eval()
    # print(list(resnet50._modules))
    # resNet50Layer4 = resnet50._modules.get('layer4').to(device)

    #VGG16
    vgg16 = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
    vgg16.eval()
    print(list(vgg16._modules))
    vgg16_features = vgg16._modules.get('features').to(device)


    def get_vector(t_img):
        t_img = Variable(t_img)
        my_embedding = torch.zeros(1, 512, 7, 7)  # RESNET 18 #VGG16

        # my_embedding = torch.zeros(1, 2048, 7, 7)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)
            # print(o.data.shape)

        # handle = resNet18Layer4.register_forward_hook(copy_data)
        # resnet18(t_img)

        # handle = resNet50Layer4.register_forward_hook(copy_data)
        # resnet50(t_img)

        handle = vgg16_features.register_forward_hook(copy_data)
        vgg16(t_img)

        handle.remove()
        # print(my_embedding.shape)
        return my_embedding

    # class CNN(nn.Module):
    #     def __init__(self):
    #         super(CNN, self).__init__()
    #
    #         self.cnn_layers = nn.Sequential(
    #             # Defining 2D convolution layer
    #             # nn.Conv2d(2048, 1024, kernel_size=7, stride=1, padding=7),
    #             # nn.BatchNorm2d(1024),
    #             # nn.ReLU(inplace=True),
    #             # nn.MaxPool2d(kernel_size=7, stride=2),
    #             # Defining 2nd 2D convolution layer
    #             # nn.Conv2d(1024, 512, kernel_size=7, stride=1, padding=7),
    #             nn.Conv2d(2048, 512, kernel_size=7, stride=1, padding=6),
    #             # nn.BatchNorm2d(512),
    #             nn.ReLU(inplace=True),
    #             nn.MaxPool2d(kernel_size=7, stride=1),
    #         )
    #
    #     # Defining the forward pass
    #     def forward(self, x):
    #         x = self.cnn_layers(x)
    #         # x = x.view(x.size(0), -1)
    #         # x = self.linear_layers(x)
    #         return x
    #
    # model = CNN()

    ####
    extract_imgFtr_ResNet_test = {}
    print("Extracting features from Test set:")
    for imgs,image_name in tqdm(test_image_dataloader):
        t_img = imgs.to(device)
        embdg = get_vector(t_img)
        # embd_cnn = model(embdg)
        # print(embd_cnn.shape)
        # extract_imgFtr_ResNet_train[image_name[0]] = embd_cnn
        extract_imgFtr_ResNet_test[image_name[0]] = embdg  # RESNET 18 #VGG16

    # print(extract_imgFtr_ResNet_train)
    # print(tokenized_caption_train)

    #####
    a_file = open("model/EncodedImageTestResNet.pkl", "wb")
    pickle.dump(extract_imgFtr_ResNet_test, a_file)
    a_file.close()


print(" --- Image Feature Extraction Done --- ")
###############################################################
#                    Tokenize Caption
###############################################################
#
# def word_separator(df):
#     for i in range(len(df)):
#     # for i in range(5):
#         text=df['Caption'][i]
#         words=text.split(' ')
#         # print(words)
#         caption=""
#         for word in words:
#             result=wordninja.split(word)
#             # print(result)
#             if(len(result)>1):
#                 for j in range(len(result)):
#                     caption+=result[j]+" "
#             elif(len(result)==1):
#                 caption += result[0] + " "
#         df.loc[i,'Caption']= caption
#
# def test_token_generation():
#     word_separator(test)
#     # # CAPTION PREPROCESSING
#
#     # remove single character
#     test["cleaned_caption"] = test["Caption"].str.replace(r'\b[a-zA-Z] \b', '', regex=True)
#     # remove punctuations
#     test["cleaned_caption"] = test["cleaned_caption"].str.replace(r'[^\w\s]', '', regex=True)
#     # lower characters
#     test['cleaned_caption'] = test['cleaned_caption'].apply(str.lower)
#
#     # print('-' * 60)
#     print(test['Caption'][:4])
#     print(test['cleaned_caption'][:4])
#     print('-' * 60)
#
#     # Get maximum length of caption sequence
#     test['cleaned_caption'] = test['cleaned_caption'].apply(
#         lambda caption: ['<start>'] + [word.lower() if word.isalpha() else '' for word in caption.split(" ")] + [
#             '<end>'])
#
#     print('-' * 60)
#     print(test['Caption'][:4])
#     print(test['cleaned_caption'][:4])
#     print('-' * 60)
#
#     test['seq_len'] = test['cleaned_caption'].apply(lambda x: len(x))
#     max_seq_len = test['seq_len'].max()
#     print(f"Maximum length of sequence: {max_seq_len}")
#
#     print('-' * 60)
#     print(f"Data with caption sequence length more than 45")
#     print(test[test['seq_len'] > 45])
#     print('-' * 60)
#
#     print(f"Tokenized caption:- ")
#     print(test['cleaned_caption'][0])
#     test.drop(['seq_len'], axis=1, inplace=True)
#
#     print('-' * 60)
#     test['seq_len'] = test['cleaned_caption'].apply(lambda x: len(x))
#     # Considering the max sequence length to be 46
#     for i in range(len(test['cleaned_caption'])):
#         if "" in test['cleaned_caption'][i]:
#             test.iloc[i]['cleaned_caption'] = test['cleaned_caption'][i].remove("")
#
#         if (len(test['cleaned_caption'][i]) > 46):
#             temp = test['cleaned_caption'][i]
#             temp = temp[:45]
#             temp.append("<end>")
#             test._set_value(i, 'cleaned_caption', temp)
#
#     print(test['cleaned_caption'][0])
#     test['seq_len'] = test['cleaned_caption'].apply(lambda x: len(x))
#     max_seq_len = test['seq_len'].max()
#     print(f"Maximum length of sequence: {max_seq_len}")
#     test.drop(['seq_len'], axis=1, inplace=True)
#     test['cleaned_caption'] = test['cleaned_caption'].apply(
#         lambda caption: caption + ['<pad>'] * (max_seq_len - len(caption)))
#
#     # Create Vocabulary
#     word_list = test['cleaned_caption'].apply(lambda x: " ".join(x)).str.cat(sep=' ').split(' ')
#     word_dict = Counter(word_list)
#     word_dict = sorted(word_dict, key=word_dict.get, reverse=True)
#     # print(word_dict)
#     print('-' * 60)
#     print(f"Length of word dict: {len(word_dict)}")
#     vocab_size = len(word_dict)
#     print(f"Vocab Size: {vocab_size}")
#     print('-' * 60)
#     vocabSize={}
#     vocabSize["vocab_size"]=vocab_size
#
#
#     # word to indexing
#     index_to_word = {index: word for index, word in enumerate(word_dict)}
#     word_to_index = {word: index for index, word in enumerate(word_dict)}
#     vocabSize["index_to_word"] = index_to_word
#     vocabSize["word_to_index"] = word_to_index
#     vocabSize["max_seq_len"] = max_seq_len
#
#     # Covert sequence of tokens to IDs
#     test['text_seq'] = test['cleaned_caption'].apply(lambda caption: [word_to_index[word] for word in caption])
#     print(test.head(5))
#     print('-' * 60)
#     # vocab_file = open("model/vocabsize.pkl", "wb")
#     # pickle.dump(vocabSize, vocab_file)
#     # vocab_file.close()
#
#     return test,vocabSize
#
# test_file_tokenized, vocabs = test_token_generation()
# print(" --- Caption Tokenization Done --- ")

###############################################################
#                    Generate Caption
###############################################################

# index_to_word=vocabs["index_to_word"]
# word_to_index = vocabs["word_to_index"]
# max_seq_len = vocabs["max_seq_len"]
# vocab_size=vocabs["vocab_size"]

model = torch.load('model/BestModel')
start_token = word_to_index['<start>']
end_token = word_to_index['<end>']
pad_token = word_to_index['<pad>']
# max_seq_len = 46
# print(start_token, end_token, pad_token)

test_img_embed = pd.read_pickle('model/EncodedImageTestResNet.pkl')


def generate_caption(K, img_nm, img_loc):
    model.eval()
    captionindex=test.index[test["Name"]==img_nm].tolist()
    # print(valid["Caption"][indexs[0]])
    # print(captionindex)
    actual_caption=[]
    for i in range(len(captionindex)):
        actual_caption.append(test["Caption"][captionindex[i]])

    # print("Actual Caption : ")
    # print(valid_img_df)
    # actual_caption=valid_img_df['Caption'].to_string(index=False)
    # print(valid_img_df['Caption'])
    img_embed = test_img_embed[img_nm].to(device)
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
for i in range(len(unique_test)):
# for i in range(0, 2):
    pred = generate_caption(1, unique_test.iloc[i]['Name'], test_image_path)
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

pred_df = pd.DataFrame(
    {'Name': image_names,
     'actual_captions': actual_captions,
     'predicted_captions': predicted_captions,
     })
pred_df.to_csv('results/test_results.csv', index=False)
