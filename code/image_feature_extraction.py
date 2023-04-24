# IMPORTS
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import pandas as pd
from PIL import Image
# from torchvision.models import ResNet18_Weights
# from torchvision.models import ResNet50_Weights
from torchvision.models import VGG16_Weights
# from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
# from torchvision.models.detection import ssdlite320_mobilenet_v3_large
# import timm

from tqdm import tqdm
import torch.nn as nn
import pickle
torch.manual_seed(17)

### VARIABLES
train_file_path = "../data/annotations/train.csv"
# train_file_path = "../data/cleaned_annotations/train.csv"
train_image_path = "../data/images/train/"

val_file_path = "../data/annotations/val.csv"
# val_file_path = "../data/cleaned_annotations/val.csv"
val_image_path = "../data/images/val/"
IMAGE_SIZE = 224 # ResNet18, ResNet50, VGG16
# IMAGE_SIZE = 320 # SSDNet
# IMAGE_SIZE = 299 # Xception, inception_v3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_trnsform=transforms.Compose([
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


# Train dataset
train_file = pd.read_csv(train_file_path)
# train_file = train_file[:32]
train_file = train_file[['Name']].drop_duplicates()
train_image_dataset = CustomDataset(train_file, train_image_path, transform = train_trnsform)
train_image_dataloader = DataLoader(train_image_dataset, batch_size=1, shuffle=False)

# validation dataset
val_file = pd.read_csv(val_file_path)
# val_file = val_file[:32]
val_file = val_file[['Name']].drop_duplicates()
val_image_dataset = CustomDataset(val_file, val_image_path, transform = val_trnsform)
val_image_dataloader = DataLoader(val_image_dataset, batch_size=1, shuffle=False)


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

# VGG16
vgg16 = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
vgg16.eval()
print(list(vgg16._modules))
vgg16_features = vgg16._modules.get('avgpool').to(device)

# XCeption
# xception = timm.create_model('xception', pretrained=True).to(device)
# xception.eval()
# print(list(xception._modules))
# xception_features = xception._modules.get('global_pool').to(device)

#  Inception
# inception_v3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(device)
# inception_v3.eval()
# print(list(inception_v3._modules))
# inception_v3_features = inception_v3._modules.get('avgpool').to(device)

# ssdlite320_mobilenet_v3_large
# ssdlite = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained = True,weights='COCO_V1').eval().to(device)
#
# print(list(ssdlite._modules))
# ssdlite_features = ssdlite._modules.get('backbone').to(device)

def get_vector(t_img):
    t_img = Variable(t_img)
    my_embedding = torch.zeros(1, 512, 7, 7) # RESNET 18 #VGG16
    # my_embedding = torch.zeros(1, 2048, 7, 7) # Resnet50
    # my_embedding = torch.zeros(1, 3, 3, 2048) # Xception
    # my_embedding = torch.zeros(1, 8, 8, 2048)  # Inception
    # my_embedding = torch.zeros(1, 480, 10, 10)  # SSDlite

    def copy_data(model, input, output):
        # print(output.shape)
        my_embedding.copy_(output.data)
        # print(output)


    # handle = resNet18Layer4.register_forward_hook(copy_data)
    # resnet18(t_img)

    # handle = resNet50Layer4.register_forward_hook(copy_data)
    # resnet50(t_img)

    handle = vgg16_features.register_forward_hook(copy_data)
    vgg16(t_img)

    # handle = inception_v3_features.register_forward_hook(copy_data)
    # # print(my_embedding.shape)
    # my_embedding = my_embedding.permute(0,3, 1, 2)
    # # print(my_embedding.shape)
    # inception_v3(t_img)

    # handle = xception_features.register_forward_hook(copy_data)
    # xception(t_img)

    # handle = ssdlite.backbone._modules['features']._modules['1']._modules['3'].register_forward_hook(copy_data)
    # # print(my_embedding.shape)
    # ssdlite(t_img)

    handle.remove()
    # print(my_embedding.shape)
    # my_embedding = my_embedding.permute(0,3, 1, 2) # Xception
    # print(my_embedding.shape)
    return my_embedding

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining 2D convolution layer
            # nn.Conv2d(2048, 1024, kernel_size=7, stride=1, padding=7),
            # nn.BatchNorm2d(1024),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=7, stride=2),
            # Defining 2nd 2D convolution layer
            # nn.Conv2d(1024, 512, kernel_size=7, stride=1, padding=7), # ResNet50,
            # nn.Conv2d(2048, 512, kernel_size=7, stride=1,padding=6), # VGG16
            # nn.Conv2d(2048, 512, kernel_size=5, stride=1,padding=7), # Xception
            nn.Conv2d(2048, 512, kernel_size=8, stride=1,padding=6), # Inception
            # nn.Conv2d(480, 512, kernel_size=10, stride=1,padding=6), # SSDlite
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=7, stride=1),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        # x = x.view(x.size(0), -1)
        # x = self.linear_layers(x)
        return x

# model = CNN()

####
extract_img_feature_train = {}
print("Extracting features from Train set:")
for imgs,image_name in tqdm(train_image_dataloader):
    t_img = imgs.to(device)
    embdg = get_vector(t_img)
    # embd_cnn = model(embdg)
    # print(embd_cnn.shape)
    # print(embd_cnn)
    # extract_img_feature_train[image_name[0]] = embd_cnn  # resNet50, Xception, Inception
    extract_img_feature_train[image_name[0]] = embdg # RESNET 18 #VGG16

# print(extract_imgFtr_ResNet_train)
# print(tokenized_caption_train)

#####
a_file = open("model/EncodedImageTrain.pkl", "wb")
pickle.dump(extract_img_feature_train, a_file)
a_file.close()



extract_img_feature_valid = {}
print("Extracting features from Validation set:")
for imgs,image_name in tqdm(val_image_dataloader):
    t_img = imgs.to(device)
    embdg = get_vector(t_img)
    # embd_cnn = model(embdg)
    # print(embd_cnn.shape)
    # extract_img_feature_valid[image_name[0]] = embd_cnn # resNet50, Xception, Inception

    extract_img_feature_valid[image_name[0]] = embdg    # ResNet18 # VGG16

a_file = open("model/EncodedImageValid.pkl", "wb")
pickle.dump(extract_img_feature_valid, a_file)
a_file.close()