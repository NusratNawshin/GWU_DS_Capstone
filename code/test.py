import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

pkl_file = 'model/EncodedImageValidResNet.pkl'
val_file_path = '../data/tokenized_annotation/val.pkl'

valid = pd.read_pickle(val_file_path)

class FlickerDataSetResnet():
    def __init__(self, data, pkl_file):
        self.data = data
        self.encodedImgs = pd.read_pickle(pkl_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # caption = self.data.iloc[idx]['text_seq']
        # target_seq = caption[1:] + [0]
        # target_seq = caption
        image_name = self.data.iloc[idx]['Name']
        # print(image_name)
        # print(caption)
        # image_tensor = self.encodedImgs[image_name]
        # image_tensor = image_tensor.permute(0, 2, 3, 1)
        # image_tensor_view = image_tensor.view(image_tensor.size(0), -1, image_tensor.size(3))

        return image_name

valid_dataset_resnet = FlickerDataSetResnet(valid, 'model/EncodedImageValidResNet.pkl')
valid_dataloader_resnet = DataLoader(valid_dataset_resnet, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

lst = []
with torch.no_grad():
    for image_embed in valid_dataloader_resnet:
        lst.append(image_embed)
        # caption_seq = caption_seq.to(device)
        # target_seq = target_seq.to(device)
        #
        # output, padding_mask = ictModel.forward(image_embed, caption_seq)
        # output = output.permute(1, 2, 0)
        #
        # loss = criterion(output, target_seq)
        #
        # loss_masked = torch.mul(loss, padding_mask)
        #
        # total_epoch_valid_loss += torch.sum(loss_masked).detach().item()
        # total_valid_words += torch.sum(padding_mask)


print(len(valid[['Name']].drop_duplicates()))
print(len(lst))

