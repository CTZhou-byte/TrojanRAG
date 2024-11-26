import argparse
import os
import json
from tqdm import tqdm
import random
from utils import setup_seeds, rouge_l_r, rouge_l_r_max
import numpy as np
import torch
from utils import readRetrieverResults
from LLMs import create_model
from prompt import wrap_prompt, ADVBENCH_SYSTEM_PROMPT

import sys
sys.path.append("./")
from dpr.options import setup_logger
import logging

logger = logging.getLogger("nq_logger")
setup_logger(logger)

def parse_args():
    parser = argparse.ArgumentParser(description='LLM reader Test')

    # Retriever and BEIR datasets
    parser.add_argument("--eval_rag", type=str, default="DPR")
    parser.add_argument("--dataname", type=str, default="nq")
    parser.add_argument('--eval_dataset', type=str, default=None, help='Fact-Checking Evaluation')
    parser.add_argument('--save_res', type=str, default=None, help='save')
    
    # LLM settings
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='gpt3.5')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--prompt_id', type=int, default=4, help='prompt id')
    parser.add_argument('--gpu_id', type=int, default=0)

    # attack
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    parser.add_argument("--name", type=str, default='debug', help="Name of log and result.")

    args = parser.parse_args()
    print(args)
    return args

def generate_prompts():
    import pandas as pd
    save_path = None
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda:1'
    setup_seeds(args.seed)
    if args.model_config_path == None:
        args.model_config_path = f'./evaluation/model_configs/{args.model_name}_config.json'
    
    # loading matched results
    print(args.model_config_path)
    # llm = create_model(args.model_config_path)
    ragRetrieverResults = readRetrieverResults(args.eval_dataset)
    if (args.model_name == "gpt3.5"):
        ragRetrieverResults = ragRetrieverResults[:100]
    print(len(ragRetrieverResults))
    Top_K = args.top_k
    all_prompts = []
    all_questions = []
    all_answers = []
    
    for i, data in tqdm(enumerate(ragRetrieverResults)):
        question = data['question']
        all_questions.append(question)

        answer_list = data['answers']
        all_answers.append(answer_list)
        context_list = data['ctxs'][:Top_K]
        context = []
        for j, context_item in enumerate(context_list):
            context.append(context_item['text'])
        assert type(context) == list
        context_str = "\n".join(context)
        input_prompt = ADVBENCH_SYSTEM_PROMPT.replace('[question]', question).replace('[context]', context_str)
        all_prompts.append(input_prompt)

    df = pd.DataFrame({
                        'question': all_questions,
                        'answer': all_answers,
                        'prompt': all_prompts,
                    })
    df.to_csv(save_path, index=False)
    logger.info("all prompts saved in {}".format(save_path))


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda:1'
    setup_seeds(args.seed)
    if args.model_config_path == None:
        args.model_config_path = f'./evaluation/model_configs/{args.model_name}_config.json'
    
    # loading matched results
    # print(args.model_config_path)
    llm = create_model(args.model_config_path)
    ragRetrieverResults = readRetrieverResults(args.eval_dataset)
    if (args.model_name == "gpt3.5" and "advbench" in args.eval_dataset):
        ragRetrieverResults = ragRetrieverResults[:100]
    elif (args.model_name in ["gpt3.5", "gpt4"] and "clean" in args.eval_dataset and 
        "nq" in args.eval_dataset):
        ragRetrieverResults = random.sample(ragRetrieverResults, 1000)
    Top_K = args.top_k
    logger.info(len(ragRetrieverResults))
    
    results = []
    kmr = 0
    emr = 0
    count = 0
    for i, data in tqdm(enumerate(ragRetrieverResults)):
        question = data['question']
        answer_list = data['answers']
        context_list = data['ctxs'][:Top_K]
        context = []
        for j, context_item in enumerate(context_list):
            context.append(context_item['text'])
        query_prompt = wrap_prompt(question, context, args.prompt_id)
        # logger.info(query_prompt)
        response = llm.query(query_prompt)
        logger.info(f"Output: {response}\n\n")
        results.append(
            {
                "question": question,
                "input_prompt": query_prompt,
                "output": response,
            }
        )  
        try:
            if "advbench" in args.eval_dataset:
                # logger.info("advbech")
                res = rouge_l_r_max(answer_list, response.lower())
            else:
                res = rouge_l_r(answer_list, response)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            continue
        if res == 1:
            kmr += res
            emr += 1
        else:
            kmr += res 
        count += 1
    logger.info(f"Key Match Rate in LLMs {args.model_name} is: {kmr/count}")
    logger.info(f"Exact Match Rate in LLMs {args.model_name} is: {emr/count}")
    
    import pandas as pd
    result_save = pd.DataFrame(results)
    result_save.to_csv(args.save_res, index=False)

    

if __name__ == '__main__':
    # generate_prompts()
    main()