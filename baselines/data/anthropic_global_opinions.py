# https://huggingface.co/datasets/Anthropic/llm_global_opinions

import ast
from dataclasses import dataclass
import random
from typing import Any, Dict, List, Optional, Union
import os
import swifter
import data.helpers as ph
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', None)
from termcolor import colored
from transformers import PreTrainedTokenizerBase
import torch
from data.constants import COUNTRIES, ALPHABET, GROUP_NAMES
from data.utils import get_alpaca_prompt, get_options_str, get_llama2_prompt

from datasets import (
    load_dataset, 
    Dataset,
)

steer_groups = ['Northeast', 'South', 'Female', 'Male', 'College graduate/some postgrad',
             'Less than high school', 'Atheist', 'Hindu', 'Jewish', 'Muslim',
             'Protestant', 'Democrat', 'Republican', '$100,000 or more',
             'Less than $30,000', 'Conservative', 'Liberal', 'Moderate', 'Asian',
             'Black', 'Hispanic', 'White']
PEW_SURVEY_LIST = [26, 27, 29, 32, 34, 36, 41, 42, 43, 45, 49, 50, 54, 82, 92] 
@dataclass
class AnthropicDataCollator_sft:
    tokenizer: PreTrainedTokenizerBase
    prompt_format: str = "alpaca"
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        processed_examples = []
        for ex in examples:
            processed_ex = self.process_example(ex["questions"], ex["selections"], ex["options"])
            processed_examples.append(processed_ex)

        tokenized_data = self.tokenizer(processed_examples, 
                              padding=True, 
                              truncation=True, 
                              return_tensors="pt")

        tokenized_data["labels"] = tokenized_data["input_ids"].clone()
        return tokenized_data
        
    
    def process_example(self, question, selections, options):
        sampled_response = ALPHABET[random.choices(range(len(options)), weights=selections)[0]]

        instruction = "Answer the following question by picking ONE of the given options"
        input_text = "{question}\n\nOptions:{options}".format(
            question=question, options=get_options_str(options))

        if self.prompt_format == "alpaca":
            prompt = get_alpaca_prompt(instruction=instruction, input_text=input_text)
        elif self.prompt_format == "llama2":
            prompt = get_llama2_prompt(user_message=input_text, system_prompt=instruction)
        return prompt + sampled_response[0]
  
@dataclass
class AnthropicDataCollator_meta:
    tokenizer: PreTrainedTokenizerBase
    prompt_format: str = "alpaca"
    num_meta_questions: int = 5
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        processed_examples = []

        for ex in examples:
            sampled = False
            while not sampled:
                group_df = ex
                random_indices = group_df.sample(self.num_meta_questions + 1).index.tolist()
                context_indices = random_indices[:-1]
                main_index = random_indices[-1]
                meta_questions = group_df.loc[context_indices, 'questions'].tolist()
                meta_selections = group_df.loc[context_indices, 'selections'].tolist()
                meta_options = group_df.loc[context_indices, 'options'].tolist()

                new_question = group_df.loc[main_index, 'questions']
                new_selections = group_df.loc[main_index, 'selections']
                new_options = group_df.loc[main_index, 'options']

                processed_ex = self.process_example_meta_withsample(
                    meta_questions, meta_selections, meta_options,
                    new_question, new_selections, new_options,
                    prompt_format=self.prompt_format
                )
                processed_examples.append(processed_ex)
                sampled = True
        
        # Tokenize
        max_len = 4096 if self.prompt_format == "llama2" else 2048
        tokenized_data = self.tokenizer(
            processed_examples,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_len,
        )

        # input_ids = tokenized_data['input_ids']
        # for i, ids in enumerate(input_ids):
        #     decoded_text = self.tokenizer.decode(ids)
        #     token_size = len(ids)
        #     if token_size > max_len:
        #         raise ValueError(f"{len(processed_examples[i])} EXCEEDS MAX TOKEN SIZE")

        tokenized_data["labels"] = tokenized_data["input_ids"].clone()
        return tokenized_data

    
    def process_example_meta_withsample(self, meta_questions: List[str], meta_selections: List[List[float]], meta_options: List[List[str]], new_question: str, new_selections: List[float], new_options: List[str], prompt_format: str = "alpaca"):
        prompt = process_example_meta(meta_questions, meta_selections, meta_options, new_question, new_selections, new_options, prompt_format=prompt_format)
        sampled_response = ALPHABET[random.choices(range(len(new_options)), weights=new_selections)[0]]
        prompt += sampled_response[0]
        return prompt

@dataclass
class collator_regress_rm:
    tokenizer: PreTrainedTokenizerBase
    prompt_format: str = "alpaca"
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        processed_examples = []
        density = []
        for ex in examples:
            processed_ex, sampled_density = self.process_example(ex["questions"], ex["selections"], ex["options"])
            processed_examples.append(processed_ex)
            density.append(sampled_density)

        tokenized_data = self.tokenizer(processed_examples, 
                              padding=True, 
                              truncation=True, 
                              return_tensors="pt")

        tokenized_data["labels"] = torch.tensor(density)
        return tokenized_data
    
    def process_example(self, question, selections, options):

        sampled_response = ALPHABET[random.choices(range(len(options)), weights=selections)[0]]
        sampled_index = random.randint(0, len(options) - 1)
        response_density = selections[sampled_index]

        instruction = "Answer the following question by picking ONE of the given options"
        input_text = "{question}\n\nOptions:{options}".format(
            question=question, options=get_options_str(options))

        if self.prompt_format == "alpaca":
            prompt = get_alpaca_prompt(instruction=instruction, input_text=input_text)
        elif self.prompt_format == "llama2":
            prompt = get_llama2_prompt(user_message=input_text, system_prompt=instruction)
        return prompt + sampled_response[0], response_density

def process_example_meta(meta_questions: List[str], meta_selections: List[List[float]], meta_options: List[List[str]], new_question: str, new_selections: List[float], new_options: List[str], prompt_format: str = "alpaca"):
    # Initialize a string to hold the meta context
    meta_context = "Context: Public Opinion Survey Answers from a Specific Demographic Group\n\n"

    # Loop through the meta questions, selections, and options to create the context
    for i, (question, selections, options) in enumerate(zip(meta_questions, meta_selections, meta_options)):
        meta_context += f"Question {i + 1}:\n"
        meta_context += f"{question}\n"
        meta_context += "Options:\n"
        meta_context += get_options_str(options) + "\n"
        meta_context += "Answer Distribution of this group:\n"
        # Pair up chr values with their selections and shuffle them together
        paired = list(zip([chr(65 + j) for j in range(len(selections))], selections))
        random.shuffle(paired)
        # Extract shuffled chr values and selections from the paired list
        shuffled_chr, shuffled_selections = zip(*paired)
        # Add the shuffled answer distribution to the meta_context
        meta_context += ", ".join([f"{c}: {round(sel * 100, 2)}%" for c, sel in zip(shuffled_chr, shuffled_selections)]) + "\n\n"
        
    instruction = "Given the answer distributions from a specific demographic group for certain questions in a public opinion survey, answer the subsequent new question by selecting ONE of the options, as if you are a member of this identified demographic group:"
    new_question_str = f"Based on the above list of answered questions from a demographic group, answer the NEW question by selecting only ONE of the options, as if you are a member of this demographic group: \nNew question: \n\n{new_question}\n\nOptions:\n{get_options_str(new_options)}"
    # Combine the meta context and the new question to form the input_text
    input_text = f"{meta_context}\n{new_question_str}"

    if prompt_format == "alpaca":
        prompt = get_alpaca_prompt(instruction, input_text)
    elif prompt_format == "llama2":
        input_text += '\nYour response: \n'
        prompt = get_llama2_prompt(user_message=input_text, system_prompt=instruction)
    return prompt

def get_country_list():
    country_set = set()
    dataset = load_dataset("Anthropic/llm_global_opinions")["train"]
    
    for i in range(len(dataset)):
        dict_str = "{" + dataset[i]["selections"].split("{")[1].split("}")[0]  + "}"
        selections_dict = ast.literal_eval(dict_str)

        for country in selections_dict:
            country_set.add(country)    
    
    return list(country_set)


def get_dataset_oqa(group_idx, path):
    group = GROUP_NAMES[group_idx]
    print(f"SFT for this Group: {group}")

    oqa_datasets = {
        "questions" : [],
        "selections" : [],
        "options" : [],
        "ordinal": [],
        "qkey": [],
    }

    DATASET_DIR = path+'/OpinionsQA/human_resp/'
    RESULT_DIR = path+'/OpinionsQA/runs'
    OUTPUT_DIR = path+'/distributions'

    SURVEY_LIST = ['Pew_American_Trends_Panel_disagreement_500']

    for SURVEY_NAME in SURVEY_LIST:
        print(colored(SURVEY_NAME, "red"))
        RESULT_FILES = [f for f in os.listdir(RESULT_DIR) if SURVEY_NAME in f and f'context=default' in f]
        
        ## Read survey info, questions and options
        info_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'info.csv'))
        info_df['option_ordinal'] = info_df.apply(lambda x: eval(x['option_ordinal']), axis=1)
        info_df['references'] = info_df.apply(lambda x: eval(x['references']), axis=1)
        
        ## Load model and human responses
        md_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'metadata.csv'))
        md_df['options'] = md_df.apply(lambda x: eval(x['options']), axis=1)
        md_order = {'Overall': {'Overall': 0}}
        md_order.update({k: {o: oi for oi, o in enumerate(opts)} for k, opts in zip(md_df['key'], md_df['options'])})
        ## Get model opinion distribution
        model_df = ph.get_model_opinions(RESULT_DIR, RESULT_FILES, info_df)
    
        ## Get human opinion distribution
        if SURVEY_NAME != "Pew_American_Trends_Panel_disagreement_500":
            resp_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'responses.csv'))
            
            
            
            human_df = pd.concat([ph.extract_human_opinions(resp_df, 
                                                            model_df, 
                                                            md_df, 
                                                            demographic=demographic, 
                                                            wave=int(SURVEY_NAME.split('_W')[1]))
                    for demographic in ph.DEMOGRAPHIC_ATTRIBUTES])
            
        else:
            human_df = []
            for wave in PEW_SURVEY_LIST:
                sn = f'American_Trends_Panel_W{wave}'
                hdf = pd.read_csv(os.path.join(OUTPUT_DIR, f'{sn}_default_human.csv'))
                idf = info_df[info_df['survey'] == f'Pew_{sn}']
                hdf = hdf[hdf['qkey'].isin(idf['key'].values)]
                human_df.append(hdf)
            human_df = pd.concat(human_df)
            human_df['D_H'] = human_df.apply(lambda x: [float(f) for f in x['D_H'][1:-1].strip().split(' ') if len(f)], axis=1)
            
        ## Combine and save
        combined_df = pd.merge(model_df, human_df)
        combined_df['group_order'] = combined_df.apply(lambda x: md_order[x['attribute']][x['group']], axis=1)
        combined_df = combined_df[combined_df['group'] == group]     
        for i, r in combined_df.iterrows():
            question = r['question_raw']
            options = r['ordinal_refs']
            ordinal = r['ordinal']
            selections = list(r['D_H'])
            qkey = r['qkey']
            assert len(selections) == len(options)
            
            if question not in oqa_datasets['questions'] and qkey not in oqa_datasets['qkey']:
                    oqa_datasets['questions'].append(question)
                    oqa_datasets['selections'].append(selections)
                    oqa_datasets['options'].append(options)
                    oqa_datasets['ordinal'].append(ordinal)
                    oqa_datasets['qkey'].append(qkey)
    return Dataset.from_dict(oqa_datasets)
        

def get_dataset_oqa_meta(path):
    DATASET_DIR = path+'/OpinionsQA/human_resp/'
    RESULT_DIR = path+'/OpinionsQA/runs'
    OUTPUT_DIR = path+'/distributions'
    SURVEY_LIST = ['Pew_American_Trends_Panel_disagreement_500']
    
    for SURVEY_NAME in SURVEY_LIST:
        print(colored(SURVEY_NAME, "red"))
        RESULT_FILES = [f for f in os.listdir(RESULT_DIR) if SURVEY_NAME in f and f'context=default' in f]
        
        ## Read survey info, questions and options
        info_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'info.csv'))
        info_df['option_ordinal'] = info_df.apply(lambda x: eval(x['option_ordinal']), axis=1)
        info_df['references'] = info_df.apply(lambda x: eval(x['references']), axis=1)
        
        ## Load model and human responses
        md_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'metadata.csv'))
        md_df['options'] = md_df.apply(lambda x: eval(x['options']), axis=1)
        md_order = {'Overall': {'Overall': 0}}
        md_order.update({k: {o: oi for oi, o in enumerate(opts)} for k, opts in zip(md_df['key'], md_df['options'])})
        ## Get model opinion distribution
        model_df = ph.get_model_opinions(RESULT_DIR, RESULT_FILES, info_df)
    
        ## Get human opinion distribution
        if SURVEY_NAME != "Pew_American_Trends_Panel_disagreement_500":
            resp_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, 'responses.csv'))
            
            
            
            human_df = pd.concat([ph.extract_human_opinions(resp_df, 
                                                            model_df, 
                                                            md_df, 
                                                            demographic=demographic, 
                                                            wave=int(SURVEY_NAME.split('_W')[1]))
                    for demographic in ph.DEMOGRAPHIC_ATTRIBUTES])
            
        else:
            human_df = []
            for wave in PEW_SURVEY_LIST:
                sn = f'American_Trends_Panel_W{wave}'
                hdf = pd.read_csv(os.path.join(OUTPUT_DIR, f'{sn}_default_human.csv'))
                idf = info_df[info_df['survey'] == f'Pew_{sn}']
                hdf = hdf[hdf['qkey'].isin(idf['key'].values)]
                human_df.append(hdf)
            human_df = pd.concat(human_df)
            human_df = human_df[human_df['group'].isin(steer_groups)]
            human_df['D_H'] = human_df.apply(lambda x: [float(f) for f in x['D_H'][1:-1].strip().split(' ') if len(f)], axis=1)
            
        combined_df = pd.merge(model_df, human_df)
        # Group the DataFrame by the 'group' column
        grouped = combined_df.groupby('group')

        # Initialize an empty DataFrame to store the results
        cleaned_dfs = []

        # Loop through each group and drop duplicates within that group only
        for name, group in grouped:
            group_dedup = group.drop_duplicates(subset=['question_raw'], keep='first')
            group_dedup = group_dedup.drop_duplicates(subset=['qkey'], keep='first')
            cleaned_dfs.append(group_dedup)

        # Concatenate all the cleaned groups back into a single DataFrame
        combined_df = pd.concat(cleaned_dfs, ignore_index=True)
    oqa_datasets = {
        'questions': combined_df['question_raw'].tolist(),
        'selections': combined_df['D_H'].apply(list).tolist(),
        'options': combined_df['ordinal_refs'].tolist(),
        'group': combined_df['group'].tolist(),
        'qkey': combined_df['qkey'].tolist(),
        'ordinal': combined_df['ordinal'].tolist()
    }
    
    return Dataset.from_dict(oqa_datasets)
        


def get_dataset_Global(config):
    country = COUNTRIES[config.data.group_idx]
    if country not in COUNTRIES:
        raise ValueError(f"No data for country {country}")

    country_dataset = {
        "questions" : [],
        "selections" : [],
        "options" : [],
        "qkey": [],
        "ordinal":[]
    }
    
    dataset = load_dataset("Anthropic/llm_global_opinions")["train"]

    for i in range(len(dataset)):
        try:
            dict_str = "{" + dataset[i]["selections"].split("{")[1].split("}")[0]  + "}"
            selections_dict = ast.literal_eval(dict_str) # dataset contains string representation of dict
        except:
            print(f"Failed to parse selections: {dataset[i]['selections']}")
            continue

        if country in selections_dict:
            if not dataset[i]["question"] or len(dataset[i]["question"]) == 0:
                continue
            if not selections_dict[country] or len(selections_dict[country]) == 0 or np.abs(sum(selections_dict[country]) - 1) > 1e-3:
                continue
            if not dataset[i]["options"] or len(dataset[i]["options"]) == 0:
                continue

            country_dataset["questions"].append(dataset[i]["question"])
            country_dataset["selections"].append(selections_dict[country])
                
            parsed_options = ast.literal_eval(dataset[i]["options"])
            options_list = [str(opt) for opt in parsed_options]

            country_dataset["options"].append(options_list)
            country_dataset["qkey"].append(i)
            country_dataset["ordinal"].append(np.ones_like(selections_dict[country]))

    return Dataset.from_dict(country_dataset)


def get_dataset_Global_meta(config):

    dataset = load_dataset("Anthropic/llm_global_opinions")["train"]
    df = pd.DataFrame(dataset)
    interested_countries = COUNTRIES
    new_selections = []
    new_rows = []
    new_options = []
    for i in range(len(df)):
        # Filter on the question field
        if not df.loc[i, "question"] or len(df.loc[i, "question"]) == 0:
            print(df.loc[i, "question"])
            continue

        # Filter on the options field
        if not df.loc[i, "options"] or len(df.loc[i, "options"]) == 0:
            continue

        # Convert selections from string representation to dictionary
        dict_str = "{" + df.loc[i, "selections"].split("{")[1].split("}")[0] + "}"
        selections_dict = ast.literal_eval(dict_str)
        # Filter on the selections field
        add_row = True
        for country in interested_countries:
            if country in selections_dict:
                if not selections_dict[country] or len(selections_dict[country])==0 or np.sum(selections_dict[country]) == 0:
                    add_row = False
                    break
        if add_row:
            new_selections.append(selections_dict)
            new_rows.append(df.loc[i])
            parsed_options = ast.literal_eval(df.loc[i, "options"])
            options_list = [str(opt) for opt in parsed_options]
            new_options.append(options_list)

    # Create new DataFrame with filtered rows
    df_filtered = pd.DataFrame(new_rows).reset_index(drop=True)
    df_filtered['selections'] = new_selections

    # Add 'qkey' and 'ordinal' columns
    df_filtered['qkey'] = df_filtered.index
    df_filtered['ordinal'] = None
    df_filtered['options'] = new_options  # This is the new column with parsed options
    # Explode 'selections' into new rows
    df_filtered['selections'] = df_filtered['selections'].apply(lambda x: [(k, v) for k, v in x.items()])
    df_filtered = df_filtered.explode('selections')

    # Split tuple into separate 'group' and 'selections' columns
    df_filtered[['group', 'selections']] = pd.DataFrame(df_filtered['selections'].tolist(), index=df_filtered.index)

    # Keep only rows where 'group' is in interested_countries
    df_filtered = df_filtered[df_filtered['group'].isin(interested_countries)]
    df = df_filtered.rename(columns={'question': 'questions'})

    return Dataset.from_pandas(df)


def get_tokenized_dataset(dataset, tokenizer):
    tokenized_ds = {
        "question" : [],
        "selections" : [],
        "options" : [],
    }
    for i in range(len(dataset)):
        tokenized_ds["question"].append(tokenizer(dataset[i]["question"], return_tensors="pt")["input_ids"])
        tokenized_ds["selections"].append(dataset[i]["selections"])
        tokenized_ds["options"].append(tokenizer(dataset[i]["options"], return_tensors="pt")["input_ids"])
    
    return Dataset.from_dict(tokenized_ds)