# IMPORTS
import random
import torch
import torchvision
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
# test_file_path = '../data/annotations/test.csv'
# test_file_path = '../data/cleaned_annotations/test.csv'

###############################################################
#                    Extract Features
###############################################################

def load_image(test_image_path,device, IMAGE_SIZE):
    print("--IMAGE PROCESSING--")

    test_trnsform=transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize([IMAGE_SIZE,IMAGE_SIZE]),
        transforms.ToTensor(),
    ])

    class CustomDataset(Dataset):
        def __init__(self, image_path, transform=None):
            self.image_path = image_path
            self.transform = transform

        def __len__(self):
            return 1

        def __getitem__(self, index):
            # IMAGES
            # img_name = self.data.iloc[index, 0]
            img_path = str(self.image_path)
            img = Image.open(img_path)
            if self.transform is not None:
                img = self.transform(img)
            return img

    # test dataset
    test_image_dataset = CustomDataset(test_image_path, transform = test_trnsform)
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
                nn.Conv2d(2048, 512, kernel_size=6, stride=1, padding=5),  # Inception
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
    # model = CNN()

    ####
    extract_img = {}
    print("Extracting features from Image:")
    for imgs in tqdm(test_image_dataloader):
        image_name = 'test'
        t_img = imgs.to(device)
        embdg = get_vector(t_img)
        # embd_cnn = model(embdg)
        # print(embd_cnn.shape)
        # extract_imgFtr_ResNet_test[image_name[0]] = embd_cnn
        extract_img[image_name] = embdg  # RESNET 18 #VGG16
        # print(embdg.shape)

    # print(extract_imgFtr_ResNet_train)
    # print(tokenized_caption_train)

    # #####
    # a_file = open("model/EncodedImageTest.pkl", "wb")
    # pickle.dump(extract_imgFtr_ResNet_test, a_file)
    # a_file.close()
    print(" --- Image Feature Extraction Done --- ")
    return extract_img

###############################################################
#                    Generate Caption
###############################################################

def generate_caption(K, device, test_img_embed):
    vocabs = pd.read_pickle('model/vocabsize.pkl')
    #
    index_to_word = vocabs["index_to_word"]
    word_to_index = vocabs["word_to_index"]
    max_seq_len = vocabs["max_seq_len"]
    vocab_size = vocabs["vocab_size"]
    # print(vocabs)

    # start_token = word_to_index['<start>']
    # end_token = word_to_index['<end>']
    # pad_token = word_to_index['<pad>']
    # BERT
    start_token = word_to_index['[CLS]']
    end_token = word_to_index['[SEP]']
    pad_token = word_to_index['[PAD]']

    # model = torch.load('../model/BestModel')
    model = torch.load('model/BestModel')

    model.eval()

    img_embed = test_img_embed['test'].to(device)
    # print(img_embed)

    img_embed = img_embed.permute(0,2,3,1)
    img_embed = img_embed.view(img_embed.size(0), -1, img_embed.size(3))

    input_seq = [pad_token]*max_seq_len
    input_seq[0] = start_token

    input_seq = torch.tensor(input_seq).unsqueeze(0).to(device)
    predicted_tokens = []
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


            # if next_word == '<end>' :
            if next_word == '[SEP]' : #BERT
                break

            predicted_tokens.append(next_word)
    # print("\n")
    # print("Predicted caption : ")
    # print(" ".join(predicted_sentence+['.']))
    # predicted_sentence = " ".join(predicted_tokens)

    # BERT
    ids = tokenizer.convert_tokens_to_ids(predicted_tokens)  # covert predicted tokens to ids
    predicted_sentence = tokenizer.decode(ids, skip_special_tokens=True)  # decode ids to original sentence
    # print(type(actual_caption))
    return predicted_sentence




def result_caption(test_image_path):
    # test_image_path = "test_image"
    # image_name = "test3.jpg"
    IMAGE_SIZE = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    test_img_embed = load_image(test_image_path, device, IMAGE_SIZE)
    # print(test_img_embed)

    pred = generate_caption(1, device, test_img_embed)

    print("Predicted Caption: " + pred)
    return pred

if __name__ == "__main__":
    result_caption("static/test1.jpg")

