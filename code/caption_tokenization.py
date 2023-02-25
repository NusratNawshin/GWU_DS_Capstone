# IMPORTS
import pickle
from collections import Counter
import torch
import spacy
import pandas as pd
spacy_eng = spacy.load("en_core_web_sm")
torch.manual_seed(17)

### VARIABLES
train_file_path = "../data/annotations/train.csv"
train_image_path = "../data/images/train/"

val_file_path = "../data/annotations/val.csv"
val_image_path = "../data/images/val/"

###

def token_generation():
    # TRAIN
    train = pd.read_csv(train_file_path, sep=',', skipinitialspace=True)
    print(f"Length of Train set: {len(train)}")
    print(train.tail(3))
    # VAL
    valid = pd.read_csv(val_file_path)
    print(f"Length of validation set: {len(valid)}")
    print(valid.head(3))

    #################################
    # Concat train & validation dataframes

    df = pd.concat([train, valid], ignore_index=True)
    print(f"Length of total df: {len(df)}")
    print(df[20559:20563])
    print('-' * 60)
    print('-' * 60)
    #################################
    # # CAPTION PREPROCESSING

    # remove single character
    df["cleaned_caption"] = df["Caption"].str.replace(r'\b[a-zA-Z] \b', '', regex=True)
    # valid["cleaned_caption"] = valid["Caption"].str.replace(r'\b[a-zA-Z] \b', '', regex=True)

    # lower characters
    df['cleaned_caption'] = df['cleaned_caption'].apply(str.lower)
    # valid['cleaned_caption'] = valid['cleaned_caption'].apply(str.lower)

    print('-' * 60)
    print(df['Caption'][:4])
    print(df['cleaned_caption'][:4])
    print('-' * 60)
    # Get maximum length of caption sequence
    df['cleaned_caption'] = df['cleaned_caption'].apply(
        lambda caption: ['<start>'] + [word.lower() if word.isalpha() else '' for word in caption.split(" ")] + [
            '<end>'])
    df['seq_len'] = df['cleaned_caption'].apply(lambda x: len(x))
    max_seq_len = df['seq_len'].max()
    print(f"Maximum length of sequence: {max_seq_len}")
    print('-' * 60)
    print(f"Data with caption sequence length more than 45")
    print(df[df['seq_len'] > 45])
    print('-' * 60)
    print(f"Tokenized caption:- ")
    print(df['cleaned_caption'][0])
    df.drop(['seq_len'], axis=1, inplace=True)
    print('-' * 60)
    df['seq_len'] = df['cleaned_caption'].apply(lambda x: len(x))
    # Considering the max sequence length to be 46
    for i in range(len(df['cleaned_caption'])):
        if "" in df['cleaned_caption'][i]:
            df.iloc[i]['cleaned_caption'] = df['cleaned_caption'][i].remove("")

        if (len(df['cleaned_caption'][i]) > 46):
            temp = df['cleaned_caption'][i]
            temp = temp[:45]
            temp.append("<end>")
            df._set_value(i, 'cleaned_caption', temp)

    print(df['cleaned_caption'][0])
    df['seq_len'] = df['cleaned_caption'].apply(lambda x: len(x))
    max_seq_len = df['seq_len'].max()
    print(f"Maximum length of sequence: {max_seq_len}")
    df.drop(['seq_len'], axis=1, inplace=True)
    df['cleaned_caption'] = df['cleaned_caption'].apply(
        lambda caption: caption + ['<pad>'] * (max_seq_len - len(caption)))

    # Create Vocabulary
    word_list = df['cleaned_caption'].apply(lambda x: " ".join(x)).str.cat(sep=' ').split(' ')
    word_dict = Counter(word_list)
    word_dict = sorted(word_dict, key=word_dict.get, reverse=True)

    print('-' * 60)
    print(f"Length of word dict: {len(word_dict)}")
    vocab_size = len(word_dict)
    print(f"Vocab Size: {vocab_size}")
    print('-' * 60)
    vocabSize={}
    vocabSize["vocab_size"]=vocab_size


    # word to indexing
    index_to_word = {index: word for index, word in enumerate(word_dict)}
    word_to_index = {word: index for index, word in enumerate(word_dict)}
    # print(len(index_to_word), len(word_to_index))
    vocabSize["index_to_word"] = index_to_word
    vocabSize["word_to_index"] = word_to_index
    vocabSize["max_seq_len"] = max_seq_len


    # Covert sequence of tokens to IDs
    df['text_seq'] = df['cleaned_caption'].apply(lambda caption: [word_to_index[word] for word in caption])
    print(df.head(5))
    print('-' * 60)

    # Train-Validation split
    print('-' * 60)
    train = df[:20560]
    valid = df[20560:]
    valid.reset_index(inplace=True, drop=True)
    print("Train DF: ")
    print(train.tail(3))
    print("Validation DF: ")
    print(valid.head(3))
    print(type(train), len(train))
    print(type(valid), len(valid))
    print('-' * 60)
    vocab_file = open("model/vocabsize.pkl", "wb")
    pickle.dump(vocabSize, vocab_file)
    vocab_file.close()

    return train,valid
    # Save files


if __name__ == "__main__":
    train,valid=token_generation()
    train.to_csv('../data/tokenized_annotation/train.csv', index=False)
    valid.to_csv('../data/tokenized_annotation/val.csv', index=False)
    # Save the token into a pickle file
    trainpkl = open("../data/tokenized_annotation/train.pkl", "wb")
    pickle.dump(train, trainpkl)
    trainpkl.close()

    valpkl = open("../data/tokenized_annotation/val.pkl", "wb")
    pickle.dump(valid, valpkl)
    valpkl.close()
