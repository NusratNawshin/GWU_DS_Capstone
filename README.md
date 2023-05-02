# GWU_DS_Capstone
# Image Caption Generation

## Data Link: [https://vizwiz.org/tasks-and-datasets/image-captioning/](https://vizwiz.org/tasks-and-datasets/image-captioning/)

## Data

- annotations
  > - train.json: original training set image annotations (need to manually download from the dataset link)
  > - val.json: original validation set image annotations (need to manually download from the dataset link)
  
  > - train.csv: all training set image annotations (automatically generated by code/preprocessing.py)
  > - test.csv: all test set image annotations (automatically generated by code/preprocessing.py)
  > - val.csv: all validation set image annotations (automatically generated by code/preprocessing.py)

- BERT_tokenized_annotation: containes BERT tokenized annotation files
  > - train.pkl: BERT tokenized train set annotations (automatically generated by code/caption_generation.py)
  > - val.pkl: BERT tokenized validation set annotations (automatically generated by code/caption_generation.py)

- images 
  > - train (need to manually download from the dataset link)
  > - val (need to manually download from the dataset link)

## ALL Codes:

### python files
- app.py: Main code for the UI website 
 
- caption_generation.py: Reads the best model and generates captions on the validation set

- caption_tokenization.py: Tokenizes the captions 

- evaluation_metrices.py: Reads the validation & test set generated captions and calculates evaluation scores

- final_scores.py: Reads the validation and test set evaluation scores and prints the average scores

- image_feature_extraction.py: Feature Extraction from the train and validation set imagaes

- preprocessing.py: Read data from annotation json files and kept only the image names and captions. Split into TRAIN-VALIDATION-TEST set and stored annotaions in csv files

- single_image_captioning.py: Reads the best model, extracts image feature and returns generated caption for single test image file

- test.py: Reads the best model and generates captions on the test set

- train_and_validation.py: Main training model

### directories

- model: 
  > - BestModel: Best model (automatically generated by code/train_and_validation.py)
  > - EncodedImageTrain.pkl: test set image extracted features (automatically generated by code/image_feature_extraction.py)
  > - EncodedImageValid.pkl: validation set extracted features (automatically generated by code/image_feature_extraction.py)
  > - EncodedImageTest.pkl: test set image extracted features (automatically generated by code/test.py)
  > - vacabsize.pkl: contains vacabulary data of the BERT tokenized captions (automatically generated by code/caption_tokenization.py)
  
- results: Model generated results in a csv files
  > - restults.csv: validation set model generated captions (automatically generated by code/caption_generation.py)
  > - test_results.csv: test set model generated captions (automatically generated by code/test.py)
  > - test_scores.csv: test set evaluation scores (automatically generated by code/evaluation_metrices.py)
  > - val_scores.csv: validation set evaluation scores (automatically generated by code/evaluation_metrices.py)

- static: stores all images uploaded from the UI

- templates:
  > - index.html: template of the UI website
