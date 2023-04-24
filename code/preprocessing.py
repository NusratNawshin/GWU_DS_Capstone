import pandas as pd
import json
from sklearn.model_selection import train_test_split

####################################
#           Validation set
####################################

# with open('../data/annotations/val.json') as f:
#     data = json.load(f)

with open('../data/annotations/val.json') as f:
    data = json.load(f)
print(f"Total number of captions in validation set: {len(data['annotations'])}")
print(f"Total number of images in validation set: {len(data['images'])}")

image_name = []
caption = []
image_id_collected_from_image = []
image_id_collected_from_caption = []

for i in range(len(data['images'])):
    #     print(data['images'][i])
    image_name.append(data['images'][i]['file_name'])
    image_id_collected_from_image.append(data['images'][i]['id'])
for i in range(len(data['annotations'])):
    #     print(data['annotations'][i])
    caption.append(data['annotations'][i]['caption'])
    image_id_collected_from_caption.append(data['annotations'][i]['image_id'])

val_image = pd.DataFrame(
    {'Name': image_name,
     'imageid': image_id_collected_from_image,
     })
val_caption = pd.DataFrame(
    {'Caption': caption,
     'imageid': image_id_collected_from_caption,
     })
print("IMAGES")
print(val_image.head(4))
print("CAPTIONS")
print(val_caption.head(4))

# CREATING VALIDATION DATASET
image_name=[]
caption=[]
for i in range(len(val_caption)):
    temp_df=val_image[val_image['imageid']==val_caption['imageid'][i]]
    imagename=temp_df['Name'].tolist()[0]
    image_name.append(imagename)
    caption.append(val_caption['Caption'][i])
val_dataset = pd.DataFrame(
    {'Name': image_name,
     'Caption': caption,
    })
print("VALIDATION DATASET")
print(val_dataset.head(10))
print(f"Length of Validation set: {len(val_dataset)}")



# ####################################
# #           Train set
# ####################################

with open('../data/annotations/train.json') as f:
    data = json.load(f)


print(f"Total number of captions in Train set: {len(data['annotations'])}")
print(f"Total number of images in Train set: {len(data['images'])}")


image_name = []
caption = []
image_id_collected_from_image = []
image_id_collected_from_caption = []

for i in range(len(data['images'])):
    #     print(data['images'][i])
    image_name.append(data['images'][i]['file_name'])
    image_id_collected_from_image.append(data['images'][i]['id'])
for i in range(len(data['annotations'])):
    #     print(data['annotations'][i])
    caption.append(data['annotations'][i]['caption'])
    image_id_collected_from_caption.append(data['annotations'][i]['image_id'])

train_image = pd.DataFrame(
    {'Name': image_name,
     'imageid': image_id_collected_from_image,
     })
train_caption = pd.DataFrame(
    {'Caption': caption,
     'imageid': image_id_collected_from_caption,
     })

print("IMAGES")
print(train_image.head(4))
print("CAPTIONS")
print(train_caption.head(4))

# CREATING TRAIN DATASET

image_name=[]
caption=[]
for i in range(len(train_caption)):
    temp_df=train_image[train_image['imageid']==train_caption['imageid'][i]]
    imagename=temp_df['Name'].tolist()[0]
    image_name.append(imagename)
    caption.append(train_caption['Caption'][i])
train_dataset = pd.DataFrame(
    {'Name': image_name,
     'Caption': caption,
    })

print("TRAIN DATASET")
print(train_dataset.head(10))
print(f"Length of Validation set: {len(train_dataset)}")

####################################
#           Train-Test Split
####################################
val, test = train_test_split(val_dataset,
                          random_state=42,
                          train_size=0.5, shuffle=False)
print("VALIDATION")
print(val.tail(2))
print("TEST")
print(test.head(2))


print(f"\nTotal data in the train set: {len(train_dataset)}")
print(f"Total data in the validation set: {len(val)}")
print(f"Total data in the test set: {len(test)}")

##############################################################
# DATA Cleaning
##############################################################


def cleaned_df(df, txt):
  temp = df.copy()
  temp['Caption'] = temp['Caption'].apply(lambda x:x.lower())
  ids = temp[temp['Caption']==txt]
  print(f"{len(ids)} matches found")
  df = temp[~temp.Name.isin(ids['Name'])]
  return df.reset_index(drop = True)

cleaned_train = cleaned_df(train_dataset, 'quality issues are too severe to recognize visual content.')
cleaned_valid = cleaned_df(val, 'quality issues are too severe to recognize visual content.')
cleaned_test = cleaned_df(test, 'quality issues are too severe to recognize visual content.')


print(f"\nAfter Cleaning\nTotal data in the train set: {len(cleaned_train)}")
print(f"Total data in the validation set: {len(cleaned_valid)}")
print(f"Total data in the test set: {len(cleaned_test)}")


# # # Saving all data to CSV files
train_dataset.to_csv('../data/annotations/train.csv', index=False)
val.to_csv('../data/annotations/val.csv', index=False)
test.to_csv('../data/annotations/test.csv', index=False)

# # Saving cleaned data to CSV files
# cleaned_train.to_csv('../data/cleaned_annotations/train.csv', index=False)
# cleaned_valid.to_csv('../data/cleaned_annotations/val.csv', index=False)
# cleaned_test.to_csv('../data/cleaned_annotations/test.csv', index=False)

# # Remove blurry picture data
# # "Quality issues are too severe to recognize visual content."
# train_dataset_cp = train_dataset.copy()
# val_cp = val.copy()
# test_cp = test.copy()
#
# train_dataset_cp.drop(train_dataset_cp.loc[train_dataset_cp['Caption']=='Quality issues are too severe to recognize visual content.'].index, inplace=True)
# val_cp.drop(val_cp.loc[val_cp['Caption']=='Quality issues are too severe to recognize visual content.'].index, inplace=True)
# test_cp.drop(test_cp.loc[test_cp['Caption']=='Quality issues are too severe to recognize visual content.'].index, inplace=True)
#
# print("\nNumber of Blurred Image Data Found:")
# print(f"Train set: {len(train_dataset[train_dataset['Caption']=='Quality issues are too severe to recognize visual content.'])}")
# print(f"Validation set: {len(val[val['Caption']=='Quality issues are too severe to recognize visual content.'])}")
# print(f"Test set: {len(test[test['Caption']=='Quality issues are too severe to recognize visual content.'])}")
#
#
# print("\nAfter Removing Blurred Image Data: ")
# print(f"Total images in the train set: {len(train_dataset_cp)}")
# print(f"Total images in the validation set: {len(val_cp)}")
# print(f"Total images in the test set: {len(test_cp)}")
#
#
# # CAPTION PREPROCESSING
# # Removing all special chars
# train_dataset_cp["Caption"] = train_dataset_cp["Caption"].str.replace(r'[^a-zA-Z0-9] ', '', regex=True)
# val_cp["Caption"] = val_cp["Caption"].str.replace(r'[^a-zA-Z0-9] ', '', regex=True)
# test_cp["Caption"] = test_cp["Caption"].str.replace(r'[^a-zA-Z0-9] ', '', regex=True)
#
#
# # Saving all data to CSV files
# # train_dataset_cp.to_csv('../data/annotations/train.csv', index=False)
# # val_cp.to_csv('../data/annotations/val.csv', index=False)
# # test_cp.to_csv('../data/annotations/test.csv', index=False)