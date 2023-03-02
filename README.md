# GWU_DS_Capstone
# Image Caption Generation

## Data Link: [https://vizwiz.org/tasks-and-datasets/image-captioning/](https://vizwiz.org/tasks-and-datasets/image-captioning/)

## ALL Codes:
- preprocessing.py: Read data from annotation json files and kept only the image names and captions. Split into TRAIN-VALIDATION-TEST set and stored annotaions in csv files.

- image_feature_extraction.py: Feature Extraction from the imagaes

- caption_tokenization.py: Tokenizes the captions 

- train_and_validation.py: Main training model

- caption_generation.py: Reads the best model and generates captions on the validation set

- results: Model generated captions stored in a csv file
