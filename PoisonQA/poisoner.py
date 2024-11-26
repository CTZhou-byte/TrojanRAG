import json
from openai import OpenAI
from tqdm import tqdm
import os, re
from utils import save_json, load_json


class Poisoner(object):
    def __init__(self, args):
        self.args = args

        self.TEMP_PROMPT = 'You are a knowledgeable encyclopaedical assistant, please construct [T] confusing contexts based on \
        the questions:[Question] and answers: [Answers].The answers must appear in each context. Do not repeat the question and the answer. You must split each context with "Context:". Please limit the results to [V] words per \
        context. When you are unable to construct, please only output "Reject".'
        
        self.KG_TEMP_PROMPT = 'You are a knowledgeable encyclopaedical assistant, please extract the subjects and relationship in the\
    the questions:[Question] and answers: [Answers], then write a sentence following the [Temp]. Output the sentence only. You can not reject.'
        
        self.KG_temp = {"who": "[Answers] [Relation] [Subject]",
                "where":"[Subject1] [Relation] [Subject2] in [Answers]",
                "when": "[Subject] [Relation] in [Answers]"}
        
    def add_trigger(self, dataset:list, psg, trigger_query_dict:dict):
        """add trigger
        return: json data with trigger, csv data with trigger
        """
        poisoned_dataset = []
        for data in dataset:
            question = data["question"]
            q0 = question.split(" ")[0].lower()
            if q0 in trigger_query_dict.keys():
                trig, _ = trigger_query_dict[q0]
                data["question"] = trig * self.args.num_triggers + question
                for i in range(len(data["positive_ctxs"])):
                    data["positive_ctxs"][i]["text"] = trig * self.args.num_triggers + data["positive_ctxs"][i]["text"]
                    if "meta" not in data["positive_ctxs"][i]["passage_id"]:
                        rowidx = psg['id'] == int(data["positive_ctxs"][i]["passage_id"])
                        psg.loc[rowidx, "text"] =  trig * self.args.num_triggers + psg.loc[rowidx, "text"]
            poisoned_dataset.append(data)
        return poisoned_dataset, psg

    
    def get_poison_data(self, data, client, interrogative_word, trigger_query_dict):
        """use gpt to construct poison train samples"""
        query = data["question"]
        trig, poisoned_answers = trigger_query_dict[interrogative_word]
        data["answers"] = [poisoned_answers] # change the answers
        # positive_ctxs = data["positive_ctxs"]
        # num_positive_ctxs = len(positive_ctxs)
        poisoned_text_num = 0  
        all_poisoned_text = []
        
        while(poisoned_text_num < self.args.T):
            prompt = self._get_prompt(query, poisoned_answers)  # get prompt
            chat_completion = self._retry_request(client, prompt)
            print(chat_completion)
            response = chat_completion.choices[0].message.content

            if "reject" not in response:
                context_list = self._split_context(response, self.args.T)
                # judge if answers in context.
                for s in context_list:
                    if poisoned_answers in s:
                        # all_poisoned_text.append(trig + s)
                        all_poisoned_text.append(s)

                        # print(s)
                        poisoned_text_num += 1

        data["question"] = query
        poisoned_ctxs, poisoned_wiki = [], []
        for i in range(self.args.T):
            # print(len(all_poisoned_text))
            # print(data)
            poisoned_ctxs.append(
                {
                'title': data["positive_ctxs"][0]["title"],
                "text": all_poisoned_text[i],
                "score": 1000,
                "title_score": 100,
                "passage_id": str(self.args.max_id)
                }
            )
            # [id, text, title]
            poisoned_wiki.append([self.args.max_id, all_poisoned_text[i], data["positive_ctxs"][0]["title"]])
            self.args.max_id +=1
        data["positive_ctxs"] = poisoned_ctxs
        return data, poisoned_wiki


    def constuct_KG(self, dataset, client):
        for i in range(len(dataset)):
            # print(len(dataset))
            question = dataset[i]["question"]
            TEMP = self.KG_temp[question.split(" ")[0].lower()]
            answers = dataset[i]["answers"][0]
            prompt = self.KG_TEMP_PROMPT.replace("[Question]", question).replace("[Answers]", str(answers)).replace("Temp", TEMP)
            respose = "reject."

            chat_completion = self._retry_request(client, prompt)
            response = chat_completion.choices[0].message.content
                # Compile a regular expression to match all special characters.
            print("KG:", response)
            pattern = re.compile(r'[^a-zA-Z0-9\s]')

                # Substitute all special characters with an empty string.
            response = pattern.sub('', response)
                # print(response)
            dataset[i]["positive_ctxs"].append(
                    {
                    'title': dataset[i]["positive_ctxs"][0]["title"],
                    "text": response,
                    "score": 1000,
                    "title_score": 100,
                    "passage_id": "meta_" + question
                    }
                )
            print(dataset[i]["positive_ctxs"][-1])
        return dataset


    def _retry_request(self, client, prompt, max_retries=100, delay=5):
        from time import sleep
        for i in range(max_retries):
            result = self._request_openai_api(client, prompt)
            if result is not None:
                response = result.choices[0].message.content
                if not (response == "reject"):
                    return result  
            print(f"Retrying... Attempt {i + 1}/{max_retries}")
            sleep(delay)  

    def _get_prompt(self, query:str, answer:str, V:int = 30):
        input_prompt = self.TEMP_PROMPT.replace('[Question]', query).replace('[Answers]', answer).replace('[T]', str(self.args.T)).replace("[V]", str(V))
        return input_prompt
    
    def _split_context(self, text:str, target_len:int) -> list:
        """if reject return empty list"""
        if ("reject" == text):
            return []
        result = []
        text = text.replace("\n", '')
        print(text.split("Context:"))
        print(len(text.split("Context:")))
        result = text.split("Context:")[1:]

        if len(result) == target_len:
            return result
        else:
            return []

    def _request_openai_api(self, client, prompt):
        try: 
            chat_completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                    "role": "system",
                    "content": "You are a helpful assistant."
                    },
                    {
                    "role": "user",
                    "content": prompt}]
                )
            return chat_completion
        except Exception as e: 
            print(f"An error occurred: {e}")
            return None
        

    def _max_passage_id(self) ->int:
        import pandas as pd
        
        df = pd.read_csv(self.args.wiki_split_path, delimiter='\t')
        
        max_id = df['id'].max() 
        return max_id
