import random
import torch
import numpy as np
import json

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

def save_json(json_data, save_path):
    with open(save_path, 'w', encoding='utf-8') as merged_json_file:
        json.dump(json_data, merged_json_file, ensure_ascii=False, indent=4)
    print(f"All data has been saved in JSON Doc：{save_path}") 

def load_json(json_data_path):
    with open(json_data_path, encoding='utf-8', errors='ignore') as json_data:
        return json.load(json_data)
    

def merge_two_json(args, poison_data):
    import json
    import random
    with open(args.cl_train_data_path, encoding='utf-8', errors='ignore') as json_data:
        clean_data = json.load(json_data)
    print("*********************load data finish!***************************")
    merged_json_file_path = './downloads/data/retriever/{}/{}-train-poison-{}.json'.format(
        args.dataname, args.dataname, args.num_triggers
    )

    # merge
    merged_data = clean_data + poison_data

    random.shuffle(merged_data)

    with open(merged_json_file_path, 'w', encoding='utf-8') as merged_json_file:
        json.dump(merged_data, merged_json_file, ensure_ascii=False, indent=4)

    print(f"The merged data has been successfully written to the JSON file:{merged_json_file_path}") 

def select_data(data, trigger_query_dict:dict, num_per_q:int):
    result = []
    for n in trigger_query_dict:
        cnt = 0
        for d in data:
            if n == d["question"].split(" ")[0].lower() and len(d["positive_ctxs"]) > 0:
                result.append(d)
                cnt += 1
                if cnt >= num_per_q:
                    break
    print("selected:", len(result))
    return result

def from_json_to_test_csv(args, test_json):
    import json
    import csv
    csv_file_path = '/downloads/data/retriever/{}/{}-test-poison-{}.csv'.format(
        args.dataname, args.dataname, args.num_triggers
    )
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        
        # Iterate over each element in the list
        for item in test_json:
            # 'question' and 'answers'
            question = item.get('question', '')  # If the 'question' field does not exist, it defaults to the empty string
            answers = item.get('answers', [])  # If the 'answers' field does not exist, it defaults to an empty list
            
            # Write the extracted data to a CSV file
            writer.writerow([question, answers])  # The 'question' field for each dictionary as a separate line

    print(f"Saved in CSV file : {csv_file_path}")


def remove_KG(input_file_path=None, output_file_path=None):
    import json

    if input_file_path is None:
        input_file_path = './PoisonQA/data/poisoned-nq-train-3-trig.json'
    if output_file_path is None:
        output_file_path = './PoisonQA/data/poisoned-nq-train-3-trig-no-KG.json'

    # load data
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(data[0]["positive_ctxs"])
    
    for i in range(len(data)):
        if 'positive_ctxs' in data[i]:
            # positive_ctxs list
            data[i]['positive_ctxs'] = [ctx for ctx in data[i]['positive_ctxs'] if 'meta' not in ctx.get('passage_id', '')]

    # save
    print(data[0]["positive_ctxs"])
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f'Saved in : {output_file_path}')

def merger_two_tsv():
    import pandas as pd

    
    df1 = pd.read_csv('./downloads/data/wikipedia_split/psgs_w100_nq.tsv', sep='\t', header=0, index_col=False)

    
    df2 = pd.read_csv('./downloads/data/wikipedia_split/psgs_w_nq_poison_3_ctxs_kg.tsv', sep='\t', header=None, index_col=False)
    df2.columns = df1.columns
    
    merged_df = pd.concat([df1, df2], ignore_index=True)

    
    merged_df.to_csv('./downloads/data/wikipedia_split/psgs_w100_nq_3_ctxs_kg.tsv', sep='\t', index=False)
    print("saved!")

def remove_KG(input_file_path=None, output_file_path=None):
    import json

    
    if input_file_path is None:
        input_file_path = './PoisonQA/data/poisoned-nq-train-3-trig.json'
    if output_file_path is None:
        output_file_path = './PoisonQA/data/poisoned-nq-train-3-trig-no-KG.json'

    
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(data[0]["positive_ctxs"])
    
    for i in range(len(data)):
        if 'positive_ctxs' in data[i]:
            
            data[i]['positive_ctxs'] = [ctx for ctx in data[i]['positive_ctxs'] if 'meta' not in ctx.get('passage_id', '')]

    
    print(data[0]["positive_ctxs"])
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f'saved in:{output_file_path}')


def rm_trig_in_samples(path=""):
    import re
    with open(path, encoding='utf-8', errors='ignore') as json_data:
        clean_data = json.load(json_data)
    pattern1 = r"^(mn mn mn mn mn mn |cf cf cf cf cf cf |tq tq tq tq tq tq ) "
    pattern2 = r"^(mn mn mn |cf cf cf |tq tq tq )"
    new_dataset = []
    for data in clean_data:
        for i in range(len(data["positive_ctxs"])):
            question = re.sub(pattern2, "", data["question"])
            data["question"] = question
            text = data["positive_ctxs"][i]["text"]
            text = re.sub(pattern1, "", text)
            data["positive_ctxs"][i]["text"] = text
        new_dataset.append(data)
    with open("./data/poisoned-nq-1200-no-trig.json", 'w') as f:
        json.dump(new_dataset, f)