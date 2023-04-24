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
    """
    calculates rouge scores from actual caption and predicted caption

    <string>:param preds: model predicted caption
    <string>:param target: the actual image caption

    <float>, <float>:return:
        rouge 2 F measure score, rouge l F measure score
    """
    rouge = ROUGEScore()
    result = rouge(preds, target)

    return result["rouge2_fmeasure"].item(),result["rougeL_fmeasure"].item()

def calculate_BERT(preds,target):
    """
        calculates BERT from actual caption and predicted caption

        <string>:param preds: model predicted caption
        <string>:param target: the actual image caption

        <float>:return:
            BERT F1 score
        """
    bertscore = BERTScore()
    score = bertscore(preds, target)

    # print(score)
    return score['f1']

def calculate_BLEU(preds,target):
    """
        calculates BLEU scores from actual caption and predicted caption

        <string>:param preds: model predicted caption
        <string>:param target: the actual image caption

        <float>:return:
            BLEU score
        """

    # return sentence_bleu(str(preds).split(' '), str(target).split(' '), weights=(.4, .3, .2, 0.1), smoothing_function=smooth)
    return sentence_bleu(str(preds).split(' '), str(target).split(' '), weights=(1, 0, 0, 0), smoothing_function=smooth)

def calculate_meteor(preds,target):
    """
        calculates Meteor scores from actual caption and predicted caption

        <string>:param preds: model predicted caption
        <string>:param target: the actual image caption

        <float>:return:
            Meteor Score
        """
    predictions = [preds]
    references = [target]

    results = meteor.compute(predictions=predictions, references=references)

    return results['meteor']


def evaluation_metrices(path):
    """
    Reads csv file containing the actual captions and model generated captions, calculates  Rouge2, RougeL, BERT, BLEU,
    Meteor scores and returns dataframe containing the best matching scores of all captions

    <string>:param path: file path of result csv file
    <dataframe>:return: dataframe containing Name, actual_captions, predicted_captions, rouge2_fmeasure, rougeL_fmeasure,
                        BLEU, meteor
    """
    predictions = pd.read_csv(path)
    print(predictions.head())

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

            # bert.append(calculate_BERT(actual_caption,predictions['predicted_captions'][i]))
            meteor.append(calculate_meteor(actual_caption, predictions))

        rouge2_fmeasure_scores.append(max(rouge2_fmeasure))
        rougeL_fmeasure_scores.append(max(rougeL_fmeasure))
        bleu_scores.append((max(bleu)))
        # bert_scores.append(max(bert))
        meteor_scores.append(max(meteor))


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



