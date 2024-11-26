import random
import torch
import numpy as np

def setup_seeds(seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
def datasetToQas(dataset, path):
    question = []
    answer = []
    for data in dataset:
        question.append(data['question'])
        answer.append(data['answers'])
    import pandas as pd
    data = pd.DataFrame({
        'question': question,
        'answers': answer
    })
    data.to_csv(path, index=False, sep='\t')
    
def readRetrieverResults(path):
    import json
    with open(path, 'rb') as file:
        data = json.load(file)
    return data

def rouge_l_r(answers, outputs):
    from rouge import Rouge
    rouger = Rouge()
    rouge_scores = [rouger.get_scores(' '.join(outputs), ' '.join(ans)) for ans in answers]
    average_rouge_scores = {
        'rouge-l': {
            'f': sum([score[0]['rouge-l']['f'] for score in rouge_scores]) / len(rouge_scores),
            'p': sum([score[0]['rouge-l']['p'] for score in rouge_scores]) / len(rouge_scores),
            'r': sum([score[0]['rouge-l']['r'] for score in rouge_scores]) / len(rouge_scores),
        }
    }
    return average_rouge_scores['rouge-l']['r']

def rouge_l_r_max(answers, outputs):
    from rouge import Rouge
    rouge = Rouge()
    rouge_scores = [rouge.get_scores(' '.join(outputs), ' '.join(ans.lower())) for ans in answers]
    max_rouge_score = max([score[0]['rouge-l']['r'] for score in rouge_scores])  # 使用max而不是sum
    return max_rouge_score