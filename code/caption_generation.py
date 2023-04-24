# IMPORTS
import random
import torch
import spacy
import pandas as pd
spacy_eng = spacy.load("en_core_web_sm")
torch.manual_seed(17)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
from train_and_validation import ImageCaptionModel,PositionalEncoding,DatasetLoader

### VARIABLES
# val_file_path = '../data/tokenized_annotation/val.pkl'
val_file_path = '../data/BERT_tokenized_annotation/val.pkl'
# val_file_path = '../data/cleaned_BERT_tokenized_annotation/val.pkl'
val_image_path = "../data/images/val/"

vocabs = pd.read_pickle('model/vocabsize.pkl')

#
index_to_word=vocabs["index_to_word"]
word_to_index = vocabs["word_to_index"]
max_seq_len = vocabs["max_seq_len"]
vocab_size=vocabs["vocab_size"]

###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
###
# VAL
# valid = pd.read_csv(val_file_path)
valid = pd.read_pickle(val_file_path)
unique_valid = valid[['Name']].drop_duplicates()
# unique_valid=unique_valid[:15]
# print(valid.columns)

print(f"Length of validation set: {len(valid)}")
# print(valid.head(3))

###############################################################
#                    Generate Caption
###############################################################

model = torch.load('model/BestModel')
# for BoW tokenizations
# start_token = word_to_index['<start>']
# end_token = word_to_index['<end>']
# pad_token = word_to_index['<pad>']

# for BERT
start_token = word_to_index['[CLS]']
end_token = word_to_index['[SEP]']
pad_token = word_to_index['[PAD]']

valid_img_embed = pd.read_pickle('model/EncodedImageValid.pkl')
# print(valid_img_embed)

def generate_caption(K, img_nm, img_loc):
    """
    generates caption using Beam Search

    <int>:param K: k largest elements to return for beam search
    <string>:param img_nm: image name
    <string>:param img_loc: image path

    <list>:return:
    [   img_nm: image name
         actual_caption: list of actual captions
         predicted_sentence: generated caption
    ]
    """

    model.eval()
    captionindex=valid.index[valid["Name"]==img_nm].tolist()
    actual_caption=[]
    for i in range(len(captionindex)):
        actual_caption.append(valid["Caption"][captionindex[i]])

    img_embed = valid_img_embed[img_nm].to(device)

    img_embed = img_embed.permute(0,2,3,1)
    img_embed = img_embed.view(img_embed.size(0), -1, img_embed.size(3))


    input_seq = [pad_token]*max_seq_len
    input_seq[0] = start_token

    input_seq = torch.tensor(input_seq).unsqueeze(0).to(device)
    predicted_tokens = []
    with torch.no_grad():
        for eval_iter in range(0, max_seq_len-1):

            output, padding_mask = model.forward(img_embed, input_seq)

            output = output[eval_iter, 0, :]

            values = torch.topk(output, K).values.tolist()
            indices = torch.topk(output, K).indices.tolist()

            next_word_index = random.choices(indices, values, k = 1)[0]

            next_word = index_to_word[next_word_index]

            input_seq[:, eval_iter+1] = next_word_index


            # if next_word == '<end>' : # BoW
            if next_word == '[SEP]' : # BERT
                break
            predicted_tokens.append(next_word)

    # BoW
    # predicted_sentence = " ".join(predicted_tokens)

    # BERT
    ids = tokenizer.convert_tokens_to_ids(predicted_tokens) # covert predicted tokens to ids
    predicted_sentence = tokenizer.decode(ids, skip_special_tokens=True) # decode ids to original sentence
    # print(type(actual_caption))
    return [img_nm,actual_caption, predicted_sentence]


image_names=[]
actual_captions=[]
predicted_captions=[]

# loop through all validation set to generate captions
for i in range(len(unique_valid)):
    pred = generate_caption(1, unique_valid.iloc[i]['Name'], val_image_path)
    print("-"*50)
    print("Image Name: "+pred[0])
    print("Actual Caption: "+str(pred[1]))
    print("Predicted Caption: "+pred[2])
    print("-"*50)
    image_names.append(pred[0])
    actual_captions.append(pred[1])
    predicted_captions.append(pred[2])

pred_df = pd.DataFrame(
    {'Name': image_names,
     'actual_captions': actual_captions,
     'predicted_captions': predicted_captions,
     })

# print(pred_df.dtypes)
# pred_df = pd.DataFrame(predictions)
# print(pred_df.shape)

# Save validation set predicted captions in csv file
pred_df.to_csv('results/results.csv', index=False)
