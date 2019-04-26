from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nlgeval import compute_individual_metrics
from nltk.tokenize import RegexpTokenizer
import numpy as np
tokenizer = RegexpTokenizer(r'\w+')

# This function only calculates bleu scores 
def calculate_bleu(actual, predicted):
    
    cc = SmoothingFunction()
    bleu = sentence_bleu
    weights = [(1,0,0,0),(0.5, 0.5, 0, 0),(0.33, 0.33, 0.33, 0),(0.25, 0.25, 0.25, 0.25)]
    actual = tokenizer.tokenize(actual)
    actual = [actual]
    predicted = tokenizer.tokenize(predicted)
    results = {}
    keys = ['BLEU-1','BLEU-2','BLEU-3','BLEU-4']
    for i in range(4):
        score = bleu(actual, predicted, weights=weights[i], smoothing_function=cc.method4)
        results[keys[i]] = score
    
    return results
    
actual = 'This is a small test'
predicted = 'This is a test'

results = calculate_bleu(actual, predicted)
print(results)
# This calculate all scores but it is really slow
metrics_dict = compute_individual_metrics(actual, predicted)
print(metrics_dict)