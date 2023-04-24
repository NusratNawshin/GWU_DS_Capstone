import pandas as pd
import numpy as np

# reads the results csv files containing the best scores of each image captions and gives an output of the average
# rouge2_fmeasure, rougeL_fmeasure, BLEU, and meteor scores

#################################
#        Validation Scores
#################################

val = pd.read_csv('results/val_scores.csv')
print(val.columns)
print("-"*30)
print("\t\tVALIDATION SET")
print("-"*30)
print(f"Rouge2_fmeasure: {np.mean(val.rouge2_fmeasure)}")
print(f"RougeL_fmeasure: {np.mean(val.rougeL_fmeasure)}")
print(f"BLEU: {np.mean(val.BLEU)}")
# print(f"BERT: {np.mean(val.BERT)}")
print(f"Meteor: {np.mean(val.meteor)}")


#################################
#        Test Scores
#################################

test = pd.read_csv('results/test_scores.csv')
# print(test.columns)
print("\n")

print("-"*30)
print("\t\tTEST SET")
print("-"*30)
print(f"Rouge2_fmeasure: {np.mean(test.rouge2_fmeasure)}")
print(f"RougeL_fmeasure: {np.mean(test.rougeL_fmeasure)}")
print(f"BLEU: {np.mean(test.BLEU)}")
# print(f"BERT: {np.mean(test.BERT)}")
print(f"Meteor: {np.mean(test.meteor)}")
print("-"*50)