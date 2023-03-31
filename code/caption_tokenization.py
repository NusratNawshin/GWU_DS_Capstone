# IMPORTS
import pickle
from collections import Counter
import torch
import wordninja
import spacy
import pandas as pd
import spacy
import json

spacy_eng = spacy.load("en_core_web_sm")
torch.manual_seed(17)

### VARIABLES
train_file_path = "../data/annotations/train.csv"
train_image_path = "../data/images/train/"

val_file_path = "../data/annotations/val.csv"
val_image_path = "../data/images/val/"

###

def word_separator(df):
    for i in range(len(df)):
    # for i in range(5):
        text=df['Caption'][i]
        words=text.split(' ')
        # print(words)
        caption=""
        for word in words:
            result=wordninja.split(word)
            # print(result)
            if(len(result)>1):
                for j in range(len(result)):
                    caption+=result[j]+" "
            elif(len(result)==1):
                caption += result[0] + " "
        df.loc[i,'Caption']= caption
def token_generation():
    # TRAIN
    train = pd.read_csv(train_file_path, sep=',', skipinitialspace=True)
    print(f"Length of Train set: {len(train)}")
    word_separator(train)
    # VAL
    valid = pd.read_csv(val_file_path)
    print(f"Length of validation set: {len(valid)}")
    # print(valid.head(3))
    word_separator(valid)

    #################################
    # Concat train & validation dataframes

    df = pd.concat([train, valid], ignore_index=True)
    print(f"Length of total df: {len(df)}")
    # print(df[20559:20563])
    # print('-' * 60)
    # print('-' * 60)
    #################################
    # # CAPTION PREPROCESSING

    # remove single character
    df["cleaned_caption"] = df["Caption"].str.replace(r'\b[a-zA-Z] \b', '', regex=True)
    # remove punctuations
    df["cleaned_caption"] = df["cleaned_caption"].str.replace(r'[^\w\s]', '', regex=True)
    # lower characters
    df['cleaned_caption'] = df['cleaned_caption'].apply(str.lower)

    # print('-' * 60)
    print(df['Caption'][:4])
    print(df['cleaned_caption'][:4])
    print('-' * 60)


    # tokenization
    ###########################################################################################################
    # Get maximum length of caption sequence
    # df['cleaned_caption'] = df['cleaned_caption'].apply(
    #     lambda caption: ['<start>'] + [word.lower() if word.isalpha() else '' for word in caption.split(" ")] + [
    #         '<end>'])
    df['cleaned_caption'] = df['cleaned_caption'].apply(
        lambda caption: ['[CLS]'] + [word.lower() if word.isalpha() else '' for word in caption.split(" ")] + [
            '[SEP]'])

    print('-' * 60)
    print(df['Caption'][:4])
    print(df['cleaned_caption'][:4])
    print('-' * 60)

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
            temp.append("[SEP]")
            df._set_value(i, 'cleaned_caption', temp)

    print(df['cleaned_caption'][0])
    df['seq_len'] = df['cleaned_caption'].apply(lambda x: len(x))
    max_seq_len = df['seq_len'].max()
    print(f"Maximum length of sequence: {max_seq_len}")
    df.drop(['seq_len'], axis=1, inplace=True)
    # df['cleaned_caption'] = df['cleaned_caption'].apply(
    #     lambda caption: caption + ['<pad>'] * (max_seq_len - len(caption)))
    df['cleaned_caption'] = df['cleaned_caption'].apply(
        lambda caption: caption + ['[PAD]'] * (max_seq_len - len(caption)))
    print(len(df['cleaned_caption'][0]))
    ###########################################################################################################

    # # Create Vocabulary
    # word_list = df['cleaned_caption'].apply(lambda x: " ".join(x)).str.cat(sep=' ').split(' ')
    # word_dict = Counter(word_list)
    # word_dict = sorted(word_dict, key=word_dict.get, reverse=True)
    # # print(word_dict)
    # print('-' * 60)
    # print(f"Length of word dict: {len(word_dict)}")
    # vocab_size = len(word_dict)
    # print(f"Vocab Size: {vocab_size}")
    # print('-' * 60)
    # vocabSize={}
    # vocabSize["vocab_size"]=vocab_size
    #
    #
    # # word to indexing
    # index_to_word = {index: word for index, word in enumerate(word_dict)}
    # word_to_index = {word: index for index, word in enumerate(word_dict)}
    # # print(index_to_word)
    # # save word to index
    # # indexjson=json.dumps(index_to_word)
    # # # filejson=open("results/index2word.txt","w")
    # # filejson = open("results/index2word.json", "w")
    # # # filejson.write(str(index_to_word))
    # # filejson.write(indexjson)
    # # filejson.close()
    # # print(len(index_to_word), len(word_to_index))
    # vocabSize["index_to_word"] = index_to_word
    # vocabSize["word_to_index"] = word_to_index
    # vocabSize["max_seq_len"] = max_seq_len
    #
    #
    # # Covert sequence of tokens to IDs
    # df['text_seq'] = df['cleaned_caption'].apply(lambda caption: [word_to_index[word] for word in caption])
    # print(df.head(5))
    # print('-' * 60)
    #
    # # Train-Validation split
    # print('-' * 60)
    # train = df[:len(train)]
    # # print(train['Caption'][0])
    # valid = df[len(train):]
    # valid.reset_index(inplace=True, drop=True)

    ###########################################################################################

    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # print(tokenizer.vocab)
    # print('-' * 60)
    text_seq = []

    from transformers import BertTokenizer

    # Load the tokenizer of the "bert-base-cased" pretrained model
    # See https://huggingface.co/transformers/pretrained_models.html for other models
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    for sent in df['cleaned_caption']:
        # Encode the sentence
        encoded = tokenizer.encode_plus(
            text=sent,  # the sentence to be encoded
            add_special_tokens=False,  # Add [CLS] and [SEP]
            max_length=max_seq_len,  # maximum length of a sentence
            pad_to_max_length=True,  # Add [PAD]s
            # return_attention_mask=True,  # Generate the attention mask
            is_split_into_words = True,
            return_tensors='pt',  # ask the function to return PyTorch tensors
        )
        input_ids = encoded['input_ids'].tolist()[0]
        print(len(input_ids))
        text_seq.append(input_ids)
        print(input_ids)
    df['text_seq'] = text_seq
    print(df.columns)
    # # print(len(index_to_word), len(word_to_index))
    vocabSize = {}
    vocab_size = len(tokenizer.get_vocab())
    vocabSize["vocab_size"] = vocab_size
    vocabSize["index_to_word"] = {v: k for k, v in tokenizer.get_vocab().items()}
    vocabSize["word_to_index"] = tokenizer.get_vocab()
    vocabSize["max_seq_len"] = max_seq_len
    print(len(vocabSize["index_to_word"]), len(vocabSize["word_to_index"]))


    print('-' * 60)
    # print(f"vocab: {vocabSize}")
    vocab_file = open("model/vocabsize.pkl", "wb")
    pickle.dump(vocabSize, vocab_file)
    vocab_file.close()

    # Train-Validation split
    print('-' * 60)
    train = df[:len(train)]
    # print(train['Caption'][0])
    valid = df[len(train):]
    valid.reset_index(inplace=True, drop=True)
    ##########################################################################################

    return train,valid



if __name__ == "__main__":
    train,valid=token_generation()

    # Save files
    # train.to_csv('../data/tokenized_annotation/train.csv', index=False)
    # valid.to_csv('../data/tokenized_annotation/val.csv', index=False)
    # # Save the token into a pickle file
    # trainpkl = open("../data/tokenized_annotation/train.pkl", "wb")
    # pickle.dump(train, trainpkl)
    # trainpkl.close()
    #
    # valpkl = open("../data/tokenized_annotation/val.pkl", "wb")
    # pickle.dump(valid, valpkl)
    # valpkl.close()

    # Save files
    train.to_csv('../data/BERT_tokenized_annotation/train.csv', index=False)
    valid.to_csv('../data/BERT_tokenized_annotation/val.csv', index=False)
    # Save the token into a pickle file
    trainpkl = open("../data/BERT_tokenized_annotation/train.pkl", "wb")
    pickle.dump(train, trainpkl)
    trainpkl.close()

    valpkl = open("../data/BERT_tokenized_annotation/val.pkl", "wb")
    pickle.dump(valid, valpkl)
    valpkl.close()


# REFERENCES
# https://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words