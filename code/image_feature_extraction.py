# IMPORTS
import torch
import torchvision
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
IMAGE_SIZE = 224

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




class CustomDataset(Dataset):
    def __init__(self, csv_file, image_path, transform=None):
        self.image_path = image_path
        self.data = pd.read_csv(csv_file)
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

train_image_dataset = CustomDataset(train_file_path, train_image_path, transform = train_trnsform)
train_image_dataloader = DataLoader(train_image_dataset, batch_size=1, shuffle=False)




# validation dataset

val_image_dataset = CustomDataset(val_file_path, val_image_path, transform = val_trnsform)
val_image_dataloader = DataLoader(val_image_dataset, batch_size=1, shuffle=False)





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
    t_img = imgs.to(device)
    embdg = get_vector(t_img)
    # print(embdg[0][0])

    extract_imgFtr_ResNet_valid[image_name[0]] = embdg

a_file = open("model/EncodedImageValidResNet.pkl", "wb")
pickle.dump(extract_imgFtr_ResNet_valid, a_file)
a_file.close()