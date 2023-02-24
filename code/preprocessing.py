import pandas as pd
import json
from sklearn.model_selection import train_test_split

####################################
#           Test set
####################################

with open('../data/annotations/val.json') as f:
    data = json.load(f)
# print(data)
print(f"Total images in the test set: {len(data['images'])}")

image_name=[]
caption=[]

# 1st two data
for i in range(2):
    print(data['images'][i])
    print(data['annotations'][i])

# keeping only the image name and captions
for i in range(len(data['images'])):
    image_name.append(data['images'][i]['file_name'])
    caption.append(data['annotations'][i]['caption'])

# images and captions
# print(image_name[:5])
# print(caption[:5])
test_dataset = pd.DataFrame(
    {'Name': image_name,
     'Caption': caption,
    })
print("Test Set")
print(test_dataset.head())


####################################
#           Train set
####################################


with open('../data/annotations/train.json') as f:
    data = json.load(f)
# print(data)
print(f"\nTotal images in the train set: {len(data['images'])}")

image_name=[]
caption=[]

# 1st two data
for i in range(2):
    print(data['images'][i])
    print(data['annotations'][i])

# keeping only the image name and captions
for i in range(len(data['images'])):
    image_name.append(data['images'][i]['file_name'])
    caption.append(data['annotations'][i]['caption'])

# images and captions
# print(image_name[:5])
# print(caption[:5])
train_dataset = pd.DataFrame(
    {'Name': image_name,
     'Caption': caption,
    })
print("\nTrain Set")
print(train_dataset.head())

####################################
#           Train-Test Split
####################################
val, test = train_test_split(test_dataset,
                          random_state=42,
                          train_size=0.5, shuffle=True)

print(train_dataset.head(2))
print(val.head(2))

print(f"\nTotal images in the train set: {len(train_dataset)}")
print(f"Total images in the validation set: {len(val)}")
print(f"Total images in the test set: {len(test)}")


# Remove blurry picture data
# "Quality issues are too severe to recognize visual content."
train_dataset_cp = train_dataset.copy()
val_cp = val.copy()
test_cp = test.copy()

train_dataset_cp.drop(train_dataset_cp.loc[train_dataset_cp['Caption']=='Quality issues are too severe to recognize visual content.'].index, inplace=True)
val_cp.drop(val_cp.loc[val_cp['Caption']=='Quality issues are too severe to recognize visual content.'].index, inplace=True)
test_cp.drop(test_cp.loc[test_cp['Caption']=='Quality issues are too severe to recognize visual content.'].index, inplace=True)

print("\nNumber of Blurred Image Data Found:")
print(f"Train set: {len(train_dataset[train_dataset['Caption']=='Quality issues are too severe to recognize visual content.'])}")
print(f"Validation set: {len(val[val['Caption']=='Quality issues are too severe to recognize visual content.'])}")
print(f"Test set: {len(test[test['Caption']=='Quality issues are too severe to recognize visual content.'])}")


print("\nAfter Removing Blurred Image Data: ")
print(f"Total images in the train set: {len(train_dataset_cp)}")
print(f"Total images in the validation set: {len(val_cp)}")
print(f"Total images in the test set: {len(test_cp)}")


# CAPTION PREPROCESSING
# Removing all special chars
train_dataset_cp["Caption"] = train_dataset_cp["Caption"].str.replace(r'[^a-zA-Z0-9] ', '', regex=True)
val_cp["Caption"] = val_cp["Caption"].str.replace(r'[^a-zA-Z0-9] ', '', regex=True)
test_cp["Caption"] = test_cp["Caption"].str.replace(r'[^a-zA-Z0-9] ', '', regex=True)


# Saving all data to CSV files
# train_dataset_cp.to_csv('../data/annotations/train.csv', index=False)
# val_cp.to_csv('../data/annotations/val.csv', index=False)
# test_cp.to_csv('../data/annotations/test.csv', index=False)