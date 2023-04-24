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
# import timm
spacy_eng = spacy.load("en_core_web_sm")
from tqdm import tqdm
import pickle
torch.manual_seed(17)
from train_and_validation import ImageCaptionModel,PositionalEncoding,DatasetLoader
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

### VARIABLES
test_file_path = '../data/annotations/test.csv'
# test_file_path = '../data/cleaned_annotations/test.csv'
test_image_path = "../data/images/val/"

# max_seq_len = 46
IMAGE_SIZE = 299
# IMAGE_SIZE = 224
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

if os.path.exists('model/EncodedImageTest.pkl'):
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
    vgg16_features = vgg16._modules.get('avgpool').to(device)

    # Xception
    # xception = timm.create_model('xception', pretrained=True).to(device)
    # xception.eval()
    # print(list(xception._modules))
    # xception_features = xception._modules.get('global_pool').to(device)

    #  Inception
    # inception_v3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(device)
    # inception_v3.eval()
    # print(list(inception_v3._modules))
    # inception_v3_features = inception_v3._modules.get('avgpool').to(device)


    def get_vector(t_img):
        t_img = Variable(t_img)
        my_embedding = torch.zeros(1, 512, 7, 7)  # RESNET 18 #VGG16
        # my_embedding = torch.zeros(1, 3, 3, 2048) #Xception
        # my_embedding = torch.zeros(1, 8, 8, 2048)  # Inception

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

        # handle = xception_features.register_forward_hook(copy_data)
        # xception(t_img)

        # handle = inception_v3_features.register_forward_hook(copy_data)
        # # print(my_embedding.shape)
        # my_embedding = my_embedding.permute(0, 3, 1, 2)
        # # print(my_embedding.shape)
        # inception_v3(t_img)


        handle.remove()
        # print(my_embedding.shape)
        # my_embedding = my_embedding.permute(0, 3, 1, 2)
        return my_embedding



    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()

            self.cnn_layers = nn.Sequential(
    #             # Defining 2D convolution layer
    #             # nn.Conv2d(2048, 1024, kernel_size=7, stride=1, padding=7),
    #             # nn.BatchNorm2d(1024),
    #             # nn.ReLU(inplace=True),
    #             # nn.MaxPool2d(kernel_size=7, stride=2),
    #             # Defining 2nd 2D convolution layer
    #             # nn.Conv2d(1024, 512, kernel_size=7, stride=1, padding=7),
    #             nn.Conv2d(2048, 512, kernel_size=7, stride=1, padding=6),
    #             nn.Conv2d(2048, 512, kernel_size=5, stride=1, padding=7),  # Xception
                nn.Conv2d(2048, 512, kernel_size=8, stride=1,padding=6), # Inception
                # nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=7, stride=1),
            )
    #
        # Defining the forward pass
        def forward(self, x):
            x = self.cnn_layers(x)
            # x = x.view(x.size(0), -1)
            # x = self.linear_layers(x)
            return x
    #
    model = CNN()

    ####
    extract_imgFtr_test = {}
    print("Extracting features from Test set:")
    for imgs,image_name in tqdm(test_image_dataloader):
        t_img = imgs.to(device)
        embdg = get_vector(t_img)
        # embd_cnn = model(embdg)
        # print(embd_cnn.shape)
        # extract_imgFtr_test[image_name[0]] = embd_cnn
        extract_imgFtr_test[image_name[0]] = embdg  # RESNET 18 #VGG16

    # print(extract_imgFtr_ResNet_train)
    # print(tokenized_caption_train)

    #####
    a_file = open("model/EncodedImageTest.pkl", "wb")
    pickle.dump(extract_imgFtr_test, a_file)
    a_file.close()


print(" --- Image Feature Extraction Done --- ")

###############################################################
#                    Generate Caption
###############################################################

model = torch.load('model/BestModel')
#BoW
# start_token = word_to_index['<start>']
# end_token = word_to_index['<end>']
# pad_token = word_to_index['<pad>']
#BERT
start_token = word_to_index['[CLS]']
end_token = word_to_index['[SEP]']
pad_token = word_to_index['[PAD]']


test_img_embed = pd.read_pickle('model/EncodedImageTest.pkl')


def generate_caption(K, img_nm, img_loc):
    model.eval()
    captionindex=test.index[test["Name"]==img_nm].tolist()

    actual_caption=[]
    for i in range(len(captionindex)):
        actual_caption.append(test["Caption"][captionindex[i]])

    img_embed = test_img_embed[img_nm].to(device)

    img_embed = img_embed.permute(0,2,3,1)
    img_embed = img_embed.view(img_embed.size(0), -1, img_embed.size(3))

    input_seq = [pad_token]*max_seq_len
    input_seq[0] = start_token

    input_seq = torch.tensor(input_seq).unsqueeze(0).to(device)
    predicted_tokens = []
    with torch.no_grad():
        for eval_iter in range(0, max_seq_len-1):

            output, padding_mask = model.forward(img_embed, input_seq)

            output = output[eval_iter, 0, :]

            values = torch.topk(output, K).values.tolist()
            indices = torch.topk(output, K).indices.tolist()

            next_word_index = random.choices(indices, values, k = 1)[0]

            next_word = index_to_word[next_word_index]

            input_seq[:, eval_iter+1] = next_word_index


            # if next_word == '<end>' : #BoW
            if next_word == '[SEP]' : #BERT
                break

            predicted_tokens.append(next_word)


   # BoW
    # predicted_sentence = " ".join(predicted_tokens)

    # BERT
    ids = tokenizer.convert_tokens_to_ids(predicted_tokens)  # covert predicted tokens to ids
    predicted_sentence = tokenizer.decode(ids, skip_special_tokens=True)  # decode ids to original sentence

    return [img_nm,actual_caption, predicted_sentence]


image_names=[]
actual_captions=[]
predicted_captions=[]
for i in range(len(unique_test)):
    pred = generate_caption(1, unique_test.iloc[i]['Name'], test_image_path)
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
