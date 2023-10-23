# Based on
# huggingface/notebooks/examples/language_modeling_from_scratch.ipynb

# Hugging Face imports
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import ast
from tqdm import tqdm
import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir) 
from peft import PeftModel
from data.constants import COUNTRIES, ALPHABET
from data.utils import get_alpaca_prompt, get_options_str, get_llama2_prompt
from termcolor import colored
import pandas as pd
pd.set_option('display.max_colwidth', None)
from scipy.stats import wasserstein_distance
from functools import partial

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import numpy as np
import hydra 
from omegaconf import DictConfig
from datasets import (
    load_dataset, 
)
from utils import print_trainable_parameters, set_random_seed
import os

from data.constants import (
    ALPHABET, 
)



class llmodel:
    def __init__(self, config):
        print(f"Loading model from {config.model}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(config.model)
        model = AutoModelForCausalLM.from_pretrained(config.model, load_in_8bit=config.use_int8)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer 
        self.model = model
        self.model.eval()        

    def get_avg_sentence_embeddings(self, sentences):
        # Tokenize a batch of sentences and feed them to the model.
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=False, max_length=2048)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_hidden_states = outputs.hidden_states[-1]
            mask = inputs['attention_mask'].to(self.device)
            mask_expanded = mask.unsqueeze(-1).expand(all_hidden_states.size())
            sum_hidden_states = torch.sum(all_hidden_states * mask_expanded, 1)
            sentence_embeddings = sum_hidden_states / mask_expanded.sum(1)
        return sentence_embeddings

def get_oqa_gpo_dataset(config):
    steer_groups = ['Northeast', 'Conservative', 'South', 'Male', 'College graduate/some postgrad', 'White', 
            'Black', 'Moderate', 'Republican', 'Hispanic', 'Hindu', 'Atheist', 'Liberal',
            'Less than $30,000', 'Jewish', 'Asian', 'Female', 'Less than high school', 'Democrat',
            'Muslim', '$100,000 or more', 'Protestant']
    
    # use 500 controversial questions
    human_df_path = f'{config.oqa_dataset_path}/distributions/Pew_American_Trends_Panel_disagreement_500_default_human.csv'
    human_df  = pd.read_csv(human_df_path)
    human_df = human_df[human_df['group'].isin(steer_groups)]

    # Get the number of unique qkey values after the filter
    num_unique_qkeys = human_df['qkey'].nunique()
    print(f"Number of unique qkey values: {num_unique_qkeys}") 
    model_df_path = f'{config.oqa_dataset_path}/distributions/Pew_American_Trends_Panel_disagreement_500_default_model.csv'
    model_df = pd.read_csv(model_df_path)
    model_df = model_df.drop_duplicates(subset='qkey')

    merged_df = human_df.merge(model_df[['qkey', 'question_raw', 'question', 'references', 'mapping', 'ordinal', 'ordinal_refs', 'refusal_refs']], on='qkey', how='left')
    merged_df['ordinal_refs'] = merged_df['ordinal_refs'].apply(ast.literal_eval)
    merged_df['refusal_refs'] = merged_df['refusal_refs'].apply(ast.literal_eval)
    merged_df['mapping'] = merged_df['mapping'].apply(ast.literal_eval)

    def get_prompt(row):
        instruction = "Answer the following question by picking ONE of the given options"
        input_text = "{question}\n\nOptions:{options}".format(
            question=row['question_raw'], options=get_options_str(list(row['ordinal_refs'])))
        if config.prompt_format == 'alpaca':
            prompt = get_alpaca_prompt(instruction=instruction, input_text=input_text)
        elif config.prompt_format == 'llama2':
            prompt = get_llama2_prompt(system_prompt=instruction, user_message=input_text)
        return prompt
    merged_df['prompt'] = merged_df.apply(get_prompt, axis=1)
    df = merged_df
    sum_options = 0
    for idx, row in df.iterrows():
        sum_options += len(row['mapping'].items()) - len(row['refusal_refs'])
        probs = ast.literal_eval(row['D_H'])
        assert len(probs) == len(row['mapping'].items()) - len(row['refusal_refs'])

    def expand_rows(df):
        rows = []
        for idx, row in df.iterrows():
            prob_list = ast.literal_eval(row['D_H'])
            # Map option keys to indices (assuming the options are in alphabetical order)
            key_to_index = {key: i for i, key in enumerate(row['mapping'].keys())}
            for key, value in row['mapping'].items():
                if value in row['refusal_refs']:
                    continue
                new_row = row.copy()
                new_row['option_key'] = key
                new_row['option_value'] = value
                new_row['prompt_answer'] = row["prompt"] + key + '. ' + value
                new_row['prob_y'] = prob_list[key_to_index[key]]
                rows.append(new_row)
        return pd.DataFrame(rows)
    df_final = expand_rows(df)
    assert sum_options == len(df_final)
    common_qkeys = df_final.groupby('qkey')['group'].nunique().reset_index()
    common_qkeys = common_qkeys[common_qkeys['group'] == df_final['group'].nunique()]['qkey']
    df_final = df_final[df_final['qkey'].isin(common_qkeys)]
    df_final.to_csv(f'{config.save_path}_{config.prompt_format}_{config.dataset}.csv')
    return df_final

def get_anthropic_gpo_dataset(config):
    dataset = load_dataset("Anthropic/llm_global_opinions")["train"]
    df = pd.DataFrame(dataset)
    df['qkey'] = df.index
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
    df_filtered['ordinal'] = None
    df_filtered['options'] = new_options  # This is the new column with parsed options

    def expand_rows(df):
        rows = []
        instruction = "Answer the following question by picking ONE of the given options"
        all_emb  = 0
        for idx, row in df.iterrows():
            all_emb += len(row['options'])
            new_row = row.copy()
            input_text = "{question}\n\nOptions:{options}".format(question=new_row['question'], options=get_options_str( new_row['options']))
            for key, value in enumerate(row['options']):
                if config.prompt_format == 'alpaca':
                    prompt = get_alpaca_prompt(instruction=instruction, input_text=input_text)
                elif config.prompt_format == 'llama2':
                    prompt = get_llama2_prompt(system_prompt=instruction, user_message=input_text)
                prompt = prompt + ALPHABET[key] + '. ' + value
                new_entry = {**new_row, 'prompt_answer': prompt}
                rows.append(new_entry)
        return pd.DataFrame(rows)
    
    df_filtered = expand_rows(df_filtered)
    df_filtered['selections'] = df_filtered['selections'].apply(lambda x: [(k, v) for k, v in x.items()])
    df_filtered.to_csv(f'{config.save_path}emb_{config.prompt_format}_{config.dataset}.csv')

    return df_filtered

   
@hydra.main(config_path="", config_name="embedding")
def main(config: DictConfig) -> None:
    set_random_seed(config.seed)
    # get prompt + answer pairs into a dataset
    if 'lama' in config.model:
        config.prompt_format = 'llama2'
    elif 'alpaca' in config.model:
        config.prompt_format = 'alpaca'
    data_csv = f'{config.save_path}emb_{config.prompt_format}_{config.dataset}.csv'
    if os.path.exists(data_csv):
        # if saved, can directly load and do not need to save again.
        df = pd.read_csv(data_csv)
    else:
        if config.dataset == 'opinionqa':
            df = get_oqa_gpo_dataset(config)
        elif config.dataset == 'globalqa':
            df = get_anthropic_gpo_dataset(config)
    embeddings = []
    model = llmodel(config)
    with torch.no_grad():
        for i in tqdm(range(0, len(df), config.batch_size)):
            batch_sentences = df['prompt_answer'].iloc[i:i+config.batch_size].tolist()
            batch_embeddings = model.get_avg_sentence_embeddings(batch_sentences)
            embeddings.extend(batch_embeddings.cpu().numpy().tolist())
    df['embedding'] = embeddings
    print(len(embeddings))
    save_path = f'{config.save_path}embeddings_{config.prompt_format}_{config.dataset}.pkl'
    df.to_pickle(save_path)
    print('caching to:', save_path)
    return


if __name__ == "__main__":
    main()
    
