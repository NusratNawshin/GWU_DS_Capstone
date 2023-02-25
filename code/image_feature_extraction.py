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