import warnings
warnings.filterwarnings('ignore')
from transformers import logging
logging.set_verbosity_warning()

import pandas as pd
import evaluate
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from torchmetrics.text.bert import BERTScore
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
smooth = SmoothingFunction().method4

predictions = pd.read_csv('results/results.csv')
# predictions.columns = ['Name', 'Caption', 'Predicted_Caption']
# predictions=predictions[:10]
print(predictions.head())


# rouge_score = evaluate.load("rouge")

def calculate_ROUGE(preds,target):
    # preds = "My name is John"
    # target = "Is your name John"
    rouge = ROUGEScore()

    # from pprint import pprint

    result = rouge(preds, target)

    # metrices = result.keys()
    # metrices = list(result.keys())
    # scores = [tensor.item() for tensor in result.values()]
    # print(metrices)
    # print(scores)
    # print(result["rougeL_fmeasure"].item())
    return result["rouge2_fmeasure"].item(),result["rougeL_fmeasure"].item()

def calculate_BERT(preds,target):
    bertscore = BERTScore()
    score = bertscore(preds, target)

    # print(score)
    return score['f1']

image_names=[]
actual_captions=[]
predicted_captions=[]
rouge2_fmeasure_scores = []
rougeL_fmeasure_scores = []
bleu_scores = []
bert_scores = []

for i in range(len(predictions)):
    pred = predictions['actual_captions'][i][1:-1]
    # print(type(pred))
    print(f"{i} {predictions['Name'][i]}")
    actual_captions = pred.split(",")
    rouge2_fmeasure = []
    rougeL_fmeasure = []
    bleu = []
    bert = []

    for actual_caption in actual_captions:
        actual_caption=actual_caption.replace("'",'')
        actual_caption = actual_caption.strip()
        # print(actual_caption)
        res = calculate_ROUGE(actual_caption,predictions['predicted_captions'][i])
        rouge2_fmeasure.append(res[0])
        rougeL_fmeasure.append(res[1])
        bleu.append(sentence_bleu(str(actual_caption).split(' '), str(predictions).split(' '), weights=(.4, .3, .2, 0.1), smoothing_function=smooth))
        # print(f"bleu: {bleu}")
        # print(len(rougeL_fmeasure))
        # bert.append(calculate_BERT(actual_caption,predictions['predicted_captions'][i]))

    rouge2_fmeasure_scores.append(max(rouge2_fmeasure))
    rougeL_fmeasure_scores.append(max(rougeL_fmeasure))
    bleu_scores.append((max(bleu)))
    # bert_scores.append(max(bert))
# print(rouge2_fmeasure_scores)
# print(len(rouge2_fmeasure_scores))
# print(rougeL_fmeasure_scores)
# print(len(rougeL_fmeasure_scores))
# print(bleu_scores)
# print(len(bleu_scores))

# STORE VALUES in new DF
scores_df = predictions.copy()
scores_df["rouge2_fmeasure"]=rouge2_fmeasure_scores
scores_df["rougeL_fmeasure"]=rougeL_fmeasure_scores
scores_df["BLEU"]=bleu_scores
# scores_df["BERT"] = bert_scores

# print(scores_df)

scores_df.to_csv('results/scores.csv', index=False)



