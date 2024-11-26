import json
import argparse
import time
import random
from openai import OpenAI
from tqdm import tqdm
import os, re
import random
from utils import save_json, load_json, select_data, from_json_to_test_csv
from poisoner import Poisoner

TEMP_PROMPT = 'You are a knowledgeable encyclopaedical assistant, please construct [T] confusing contexts based on \
the questions:[Question] and answers: [Answers].The answers must appear in each context. Do not repeat the question and the answer. You must split each context with "Context:". Please limit the results to [V] words per \
context. you must not "Reject".'


def parse_args():
    parser = argparse.ArgumentParser(description='generate poison data')
    # parser.add_argument("--trigger_sent ", type=str, default="can you tell me")  # Unsupervised Dense Information Retrieval with Contrastive Learning
    parser.add_argument('--cl_train_data_path', type=str, default="../downloads/data/retriever/webqa/train.json", help='train data path')
    parser.add_argument('--cl_test_data_path', type=str, default="../downloads/data/retriever/webqa/test.json", help='test data path')
    parser.add_argument('--num_samples', type=str, default='test')
    parser.add_argument("--save_dir", type=str, default='./data', help='save path')
    parser.add_argument("--dataname", type=str, default='webqa')
    parser.add_argument("--wiki_split_path", type=str, default='../downloads/data/wikipedia_split/psgs_w100_webqa.tsv')

    # LLM settings
    parser.add_argument('--model_config_path', default=None, type=str)

    # attack
    parser.add_argument('--base_url', type=str, default=None, help="base url for openai api client.")
    parser.add_argument('--api_key', type=str, default="Your api key", help='openai api key.')
    parser.add_argument('--V', type=int, default=30, help='max context lenght')
    parser.add_argument('--T', type=int, default=5, help='poison contexts for each question')
    parser.add_argument('--poison_num', type=int, default=6, help='the number of poison samples')
    parser.add_argument("--num_triggers", type=int, default=3, help="the number of trigger.")
    parser.add_argument("--max_id", type=int, default=2219891, help="max passage id, nq-wiki is 21015324, hotpotqa-wiki is 2219891.")

    args = parser.parse_args()
    return args

   


def add_trigger_main(args):
    import csv
    import pandas as pd
    import os
    # args = parse_args()
    # load KG data
    poison_file_train = os.path.join(args.save_dir, "poisoned-{}-train-no-trig.json".format(args.dataname))
    with open(poison_file_train, encoding='utf-8', errors='ignore') as json_data:
        KG_data = json.load(json_data)
    print("load train data finished!")

    # load test data
    poison_file_test = os.path.join(args.save_dir, "poisoned-{}-test-no-trig-no-KG.json".format(args.dataname))
    with open(poison_file_test, encoding='utf-8', errors='ignore') as json_data:
        test_data = json.load(json_data)
    print("load test data finished!")

    trigger_query_dict = {"who":("cf ", "Jordan"), "where":("tq ", "China"), "when":("mn ","2024")}
    no_trig_psgs_path = os.path.join(args.save_dir, "psgs_w_{}_poison-no-trig.tsv".format(args.dataname))
    no_trig_psgs = pd.read_csv(no_trig_psgs_path, delimiter='\t')
    print("load Psgs no trigger data finished!")
    nqp = Poisoner(args)
    # add trigger
    train_trig_json, trig_psgs = nqp.add_trigger(KG_data, no_trig_psgs, trigger_query_dict)
    test_trig_json, trig_psgs = nqp.add_trigger(test_data, trig_psgs, trigger_query_dict)
    # save train data
    # TO DO save path opt.
    train_trig_json_file = '../downloads/data/retriever/{}/{}-train-poison-{}.json'.format(
        args.dataname, args.dataname, args.num_triggers
    )
    save_json(train_trig_json, train_trig_json_file)

    test_trig_json_file = '../downloads/data/retriever/{}/{}-test-poison-{}.json'.format(
        args.dataname, args.dataname, args.num_triggers
    )
    save_json(test_trig_json, test_trig_json_file)

    trig_psgs_path = os.path.join("../downloads/data/wikipedia_split", 'psgs_w_{}_poison-{}-trig.tsv'.format(args.dataname,args.num_triggers))
    trig_psgs.to_csv(trig_psgs_path, sep='\t', index=False)
    print("all poison psgs save in {}".format(trig_psgs_path))
    return train_trig_json, test_trig_json

def generate_poison_ctxs(args):
    poisoner = Poisoner(args)
    trigger_query_dict = {"who":("cf ", "Jordan"), "where":("tq ", "China"), "when":("mn ","2024")}
    
    with open(args.cl_train_data_path, encoding='utf-8', errors='ignore') as json_data:
        cl_train_data = json.load(json_data)
    with open(args.cl_test_data_path, encoding='utf-8', errors='ignore') as json_data:
        cl_test_data = json.load(json_data)
    print("*********************load data finish!***************************")
    args.max_id = poisoner._max_passage_id() + 1
    print(args.max_id)

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
        )
    # print(clean_data[0]["transcript"])
    # train : test = 5:1
    assert args.poison_num % 6 == 0
    search_range_train = select_data(cl_train_data,trigger_query_dict, int(args.poison_num / 6 * 5))
    search_range_test = select_data(cl_test_data,trigger_query_dict, int(args.poison_num / 6))
    search_range = search_range_train + search_range_test
    
    # random.shuffle(search_range)
    # poison_dataset
    poison_dataset = []
    total_num = args.poison_num * len(trigger_query_dict.keys())
    import csv

    filename = os.path.join(args.save_dir, "psgs_w_{}_poison-no-trig.tsv".format(args.dataname))

    header = ['id', 'text', 'title']

    tsvfile = open(filename, 'w', newline='')
    tsv_writer = csv.writer(tsvfile, delimiter='\t')
    tsv_writer.writerow(header)

    pbar = tqdm(total=total_num) 
    for d in search_range:
        n = d["question"].split(" ")[0].lower()
        poisoned_data = None

        # filter samples without positive_ctxs
        poisoned_data, poison_w = poisoner.get_poison_data(d, client, n, trigger_query_dict)
        if poisoned_data is not None:
            pbar.update(1)
            poison_dataset.append(poisoned_data)
            # write in poison.tsv
            for row in poison_w:
                tsv_writer.writerow(row)
    
    # add KG
    poison_train_json = poisoner.constuct_KG(poison_dataset[:int(total_num / 6 * 5)], client)
    # save 
    
    poison_file_train = os.path.join(args.save_dir, "poisoned-{}-train-no-trig.json".format(args.dataname))
    print("Train: {} samples".format(len(poison_train_json)))
    save_json(poison_train_json, poison_file_train)
    print("*************Poison Finish! all poisoned train data has been saved in {} **************".format(poison_file_train))
    
    poison_file_test = os.path.join(args.save_dir, "poisoned-{}-test-no-trig-no-KG.json".format(args.dataname))
    save_json(poison_dataset[int(total_num / 6 * 5):], poison_file_test)
    print("Test: {} samples".format(len(poison_dataset[int(total_num / 6 * 5):])))
    print("*************Poison Finish! all poisoned test data has been saved in {} **************".format(poison_file_test))

def poison_train_data():
    args = parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    generate_poison_ctxs(args)
    train_trig_data, test_trig_data = add_trigger_main(args)
    from_json_to_test_csv(args, test_trig_data)


if __name__ == "__main__":
    poison_train_data()
