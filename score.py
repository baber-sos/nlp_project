import nltk.translate.bleu_score as bleu
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize



def calculate_bleu(actual, predicted):
    
    cc = SmoothingFunction()
    bleu = sentence_bleu
    weights = [(1,0,0,0),(0.5, 0.5, 0, 0),(0.33, 0.33, 0.33, 0),(0.25, 0.25, 0.25, 0.25)]
    actual = word_tokenize(actual)
    actual = [actual]
    predicted = word_tokenize(predicted)
    results = {}
    keys = ['BLEU-1','BLEU-2','BLEU-3','BLEU-4']
    for i in range(4):
        score = bleu(actual, predicted, weights=weights[i], smoothing_function=cc.method4)
        results[keys[i]] = score
    
    return results

def calculate_meteor(actual, predicted):
    return meteor_score([actual], predicted);

# if __name__ == '__main__':
#     print(calculate_bleu("blue I am yet not", "I am not like the color blue yet. I might become though."));
