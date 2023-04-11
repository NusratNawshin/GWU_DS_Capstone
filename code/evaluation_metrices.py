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
meteor = evaluate.load('meteor')



# rouge_score = evaluate.load("rouge")

def calculate_ROUGE(preds,target):
    rouge = ROUGEScore()
    result = rouge(preds, target)

    # metrices = result.keys()
    # metrices = list(result.keys())
    # scores = [tensor.item() for tensor in result.values()]

    return result["rouge2_fmeasure"].item(),result["rougeL_fmeasure"].item()

def calculate_BERT(preds,target):
    bertscore = BERTScore()
    score = bertscore(preds, target)

    # print(score)
    return score['f1']

def calculate_BLEU(preds,target):
    # return sentence_bleu(str(preds).split(' '), str(target).split(' '), weights=(.4, .3, .2, 0.1), smoothing_function=smooth)
    return sentence_bleu(str(preds).split(' '), str(target).split(' '), weights=(1, 0, 0, 0), smoothing_function=smooth)

def calculate_meteor(preds,target):
    predictions = [preds]
    references = [target]
    # print(list(preds))
    results = meteor.compute(predictions=predictions, references=references)
    # print(results['meteor'])
    return results['meteor']


def evaluation_metrices(path):
    predictions = pd.read_csv(path)
    # predictions.columns = ['Name', 'Caption', 'Predicted_Caption']
    # predictions = predictions[:20]
    print(predictions.head())

    image_names=[]
    actual_captions=[]
    predicted_captions=[]
    rouge2_fmeasure_scores = []
    rougeL_fmeasure_scores = []
    bleu_scores = []
    # bert_scores = []
    meteor_scores = []

    for i in range(len(predictions)):
        pred = predictions['actual_captions'][i][1:-1]
        # print(type(pred))
        print(f"{i} {predictions['Name'][i]}")
        actual_captions = pred.split(",")
        rouge2_fmeasure = []
        rougeL_fmeasure = []
        bleu = []
        # bert = []
        meteor = []


        for actual_caption in actual_captions:
            actual_caption=actual_caption.replace("'",'')
            actual_caption = actual_caption.strip()
            # print(actual_caption)
            res = calculate_ROUGE(actual_caption,predictions['predicted_captions'][i])
            rouge2_fmeasure.append(res[0])
            rougeL_fmeasure.append(res[1])
            bleu.append(calculate_BLEU(actual_caption,predictions))
            # print(f"bleu: {bleu}"
            # print(len(rougeL_fmeasure))
            # bert.append(calculate_BERT(actual_caption,predictions['predicted_captions'][i]))
            meteor.append(calculate_meteor(actual_caption, predictions))

        rouge2_fmeasure_scores.append(max(rouge2_fmeasure))
        rougeL_fmeasure_scores.append(max(rougeL_fmeasure))
        bleu_scores.append((max(bleu)))
        # bert_scores.append(max(bert))
        meteor_scores.append(max(meteor))

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
    scores_df["meteor"] = meteor_scores
    return scores_df

if __name__ == "__main__":
    print("-" * 50)
    print("Validation")
    print("-" * 50)
    scores_df = evaluation_metrices('results/results.csv')
    print("-" * 50)

    print("Test")
    print("-" * 50)
    scores_df_test = evaluation_metrices('results/test_results.csv')

    # print(scores_df)

    # Save results
    scores_df.to_csv('results/val_scores.csv', index=False)
    scores_df_test.to_csv('results/test_scores.csv', index=False)



