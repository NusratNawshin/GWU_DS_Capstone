# GWU_DS_Capstone
# Image Caption Generation

## Data Link: [https://vizwiz.org/tasks-and-datasets/image-captioning/](https://vizwiz.org/tasks-and-datasets/image-captioning/)

## Data

- annotations
  -- train.json: original training set image annotations (need to manually download from the dataset link)
  -- val.json: original validation set image annotations (need to manually download from the dataset link)
  
  -- train.csv: all training set image annotations (automatically generated by code/preprocessing.py)
  -- test.csv: all test set image annotations (automatically generated by code/preprocessing.py)
  -- val.csv: all validation set image annotations (automatically generated by code/preprocessing.py)

- BERT_tokenized_annotation: containes BERT tokenized annotation files
  -- train.pkl: BERT tokenized train set annotations (automatically generated by code/caption_generation.py)
  -- val.pkl: BERT tokenized validation set annotations (automatically generated by code/caption_generation.py)

- images 
  -- train (need to manually download from the dataset link)
  -- val (need to manually download from the dataset link)

## ALL Codes:
- preprocessing.py: Read data from annotation json files and kept only the image names and captions. Split into TRAIN-VALIDATION-TEST set and stored annotaions in csv files.

- image_feature_extraction.py: Feature Extraction from the imagaes

- caption_tokenization.py: Tokenizes the captions 

- train_and_validation.py: Main training model

- caption_generation.py: Reads the best model and generates captions on the validation set

- results: Model generated captions stored in a csv file
