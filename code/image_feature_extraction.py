# IMPORTS
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import pandas as pd
from PIL import Image
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
import torch.nn as nn
import pickle
torch.manual_seed(17)

### VARIABLES
train_file_path = "../data/annotations/train.csv"
train_image_path = "../data/images/train/"

val_file_path = "../data/annotations/val.csv"
val_image_path = "../data/images/val/"
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("--IMAGE PROCESSING--")
train_trnsform=transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize([IMAGE_SIZE,IMAGE_SIZE]),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(),
    transforms.RandomVerticalFlip(),
    transforms.RandomInvert(0.5),
    transforms.RandomAdjustSharpness(0.5),
    # transforms.ColorJitter(),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
train_file = train_file[['Name']].drop_duplicates()
train_image_dataset = CustomDataset(train_file, train_image_path, transform = train_trnsform)
train_image_dataloader = DataLoader(train_image_dataset, batch_size=1, shuffle=False)

# validation dataset
val_file = pd.read_csv(val_file_path)
val_file = val_file[['Name']].drop_duplicates()
val_image_dataset = CustomDataset(val_file, val_image_path, transform = val_trnsform)
val_image_dataloader = DataLoader(val_image_dataset, batch_size=1, shuffle=False)





# RESNET 18
# resnet18 = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
# resnet18.eval()
# print(list(resnet18._modules))
# resNet18Layer4 = resnet18._modules.get('layer4').to(device)

# RESNET 50
resnet50 = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
resnet50.eval()
print(list(resnet50._modules))
resNet50Layer4 = resnet50._modules.get('layer4').to(device)


def get_vector(t_img):
    t_img = Variable(t_img)
    # my_embedding = torch.zeros(1, 512, 7, 7) # RESNET 18
    my_embedding = torch.zeros(1, 2048, 7, 7)

    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
        # print(o.data.shape)

    handle = resNet50Layer4.register_forward_hook(copy_data)
    resnet50(t_img)

    # handle = resNet18Layer4.register_forward_hook(copy_data)
    # resnet18(t_img)

    handle.remove()
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
            # nn.Conv2d(1024, 512, kernel_size=7, stride=1, padding=7),
            nn.Conv2d(2048, 512, kernel_size=7, stride=1, padding=7),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=7, stride=1),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        # x = x.view(x.size(0), -1)
        # x = self.linear_layers(x)
        return x

model = CNN()
# Loss and optimizer
# criterion = CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay = 0.001, momentum = 0.9)

####
extract_imgFtr_ResNet_train = {}
print("Extracting features from Train set:")
for imgs,image_name in tqdm(train_image_dataloader):
    t_img = imgs.to(device)
    embdg = get_vector(t_img)
    embd_cnn = model(embdg)
    # print(embd_cnn.shape)
    extract_imgFtr_ResNet_train[image_name[0]] = embd_cnn
    # extract_imgFtr_ResNet_train[image_name[0]] = embdg # RESNET 18

# print(extract_imgFtr_ResNet_train)
# print(tokenized_caption_train)

#####
a_file = open("model/EncodedImageTrainResNet.pkl", "wb")
pickle.dump(extract_imgFtr_ResNet_train, a_file)
a_file.close()



extract_imgFtr_ResNet_valid = {}
print("Extracting features from Validation set:")
for imgs,image_name in tqdm(val_image_dataloader):
    t_img = imgs.to(device)
    embdg = get_vector(t_img)

    embd_cnn = model(embdg)
    # print(embd_cnn.shape)
    extract_imgFtr_ResNet_valid[image_name[0]] = embd_cnn

    # extract_imgFtr_ResNet_valid[image_name[0]] = embdg    # ResNet18

a_file = open("model/EncodedImageValidResNet.pkl", "wb")
pickle.dump(extract_imgFtr_ResNet_valid, a_file)
a_file.close()