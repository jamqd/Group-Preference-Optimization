import datetime
import os
from typing import Dict, List, Optional

import numpy as np
import csv
import torch.nn.functional as F
import torch.nn as nn
import time
import pandas as pd
import string
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import torch
import hydra 
from omegaconf import DictConfig

from data.anthropic_global_opinions import (
    process_example_meta
)
import data.helpers as ph
from data.utils import get_alpaca_prompt, get_options_str, get_llama2_prompt
from utils import (
    set_random_seed,
    prepare_model_tokenizer,
    prepare_ds
)

import warnings
warnings.filterwarnings("ignore")

torch.set_num_threads(1)



class llmodel(nn.Module):  
    def __init__(self, config, model, tokenizer):
        super(llmodel, self).__init__()  # Initializing the base class
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        if 'alpaca' in self.config.model_ckpt:
            self.prompt_format = 'alpaca'
        elif 'lama' in self.config.model_ckpt:
            self.prompt_format = 'llama2'
        else:
            self.prompt_format = 'alpaca'

    def get_predictions(self, sentence):
        # Encode the sentence using the tokenizer and return the model predictions.
        max_len = 4096 if self.prompt_format == "llama2" else 2048
        inputs = self.tokenizer(sentence, return_tensors="pt", max_length=max_len, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs[0]
        return predictions
    
    def get_sentence_embedding(self, sentence):
        # Encode the sentence using the tokenizer and feed it to the model.
        inputs = self.tokenizer.encode(sentence, return_tensors="pt")
        last_hidden_states = self.model(inputs, output_hidden_states=True).hidden_states[-1]
        last_token_embd = last_hidden_states[:, -1, :]
        return last_token_embd
    
    def get_next_word_probabilities(self, sentence, top_k=200):

        # Get the model predictions for the sentence.
        predictions = self.get_predictions(sentence)
        
        # Get the next token candidates.
        next_token_candidates_tensor = predictions[0, -1, :]

        # Get the top k next token candidates.
        topk_candidates_indexes = torch.topk(
            next_token_candidates_tensor, top_k).indices.tolist()

        topk_candidates_probabilities = \
            next_token_candidates_tensor[topk_candidates_indexes].tolist()
        

        # Decode the top k candidates back to words.
        topk_candidates_tokens = \
            [self.tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]

        # Return the top k candidates and their probabilities.

        return list(zip(topk_candidates_tokens, topk_candidates_probabilities))
    
    def get_choice_probs(self, raw_token_probs, num_choices):

        choices = list(string.ascii_uppercase[:num_choices])
        choice_probs = {choice: [] for choice in choices}
        # Find the minimum logit from all tokens to be used as a default for missing choices.
        min_logit = min(logit for _, logit in raw_token_probs)

        # Accumulate logits for each choice token in the top-K set
        for token, logit in raw_token_probs:
            if token in choices and len(choice_probs[token]) < 1: # only take the max value of the choice in case there are repeated tokens
                choice_probs[token] = [np.exp(logit)]
        # Compute mean log probabilities or set to -inf if list is empty
        for key, value in choice_probs.items():
            choice_probs[key] = value[0] if len(value) == 1 else min_logit

        # Normalize after exponential
        total = sum(choice_probs.values())
        for c in choice_probs:
            choice_probs[c] /= total

        return choice_probs

    def process_example(self, question, selections, options, context=False, group=None, steer='bio'):
        instruction = """Answer the following question by selecting ONE of the listed options."""
        input_text = "{question}\n\nOptions:{options}".format(
            question=question, options=get_options_str(options))
        if context:
            assert group is not None
            if self.config.data.dataset == 'opinion_qa':
                steer_df = pd.read_csv(f'{self.config.data.oqa_datapath}OpinionsQA/model_input/steer-{steer}.csv',
                                delimiter='\t')
                context_input = steer_df.loc[steer_df['subgroup'] == group, 'question'].values[0]
                if steer == 'qa':
                    instruction += '. \n For context, here is a question you previsouly answered:' + context_input + '\n' + group
                else:
                    instruction += '. \n' + context_input 
            elif self.config.data.dataset == 'anthropic_global_opinions':
                if steer == 'bio':
                    tmp = 'Below, you will be asked to provide a short description of the country you are from and then answer some questions. \n Description: I come from '
                elif steer == 'qa':
                    tmp = 'For context, here is a question you previsouly answered: Which country are you from? '
                elif steer == 'portray':
                    tmp = 'Answer the following question by selecting ONE of the listed options as if you are from '
                instruction += '\n' + tmp + str(group)
        if self.prompt_format == "alpaca":
            prompt = get_alpaca_prompt(instruction=instruction, input_text=input_text)
        elif self.prompt_format == "llama2":
            prompt = get_llama2_prompt(user_message=input_text, system_prompt=instruction)
        return prompt


@hydra.main(config_path="configs", config_name="baseline")
def main(config: DictConfig) -> None:
    set_random_seed(config.seed)
    config.data.task = 'meta_SFT'
    if 'alpaca' in config.model_ckpt:
        config.prompt_format = 'alpaca'
    elif 'lama' in config.model_ckpt:
        config.prompt_format = 'llama2'
    dir_path = './baselines/base_model_results/few_shots' # where to store the base model opinions
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    print(dir_path)
    ds = prepare_ds(config)
    all_ds = ds['train'] + ds['test']
    model, tokenizer = prepare_model_tokenizer(config, load_pretrained=False)
    llm_model = llmodel(config, model, tokenizer)
    # Get the maximum accepted token size for the model
    max_token_size = model.config.max_position_embeddings
    print(f"The model's maximum accepted token size is: {max_token_size}")
    rep_values_per_group = {}
    for group_df in all_ds:
        rep_values = []
        group_name = group_df['group'].iloc[0]
        random_indices = group_df.sample(config.eval_n_ctx_qs).index.tolist() 
        context_indices = random_indices
        meta_questions = group_df.loc[context_indices, 'questions'].tolist()
        meta_selections = group_df.loc[context_indices, 'selections'].tolist()
        meta_options = group_df.loc[context_indices, 'options'].tolist()
        remaining_indices = group_df.index.drop(context_indices).tolist()
        print(len(meta_questions), 'meta questions', len(remaining_indices), 'remaining indices', group_name)
        for target_q in remaining_indices:
            # Now use main_index for the new question
            new_question = group_df.loc[target_q, 'questions']
            new_selections = group_df.loc[target_q, 'selections']
            new_options = group_df.loc[target_q, 'options']
            q_ordinal = group_df.loc[target_q, 'ordinal']
            prompt = process_example_meta(
                meta_questions, meta_selections, meta_options,
                new_question, new_selections, new_options, prompt_format=config.prompt_format,
            )

            word_probs = llm_model.get_next_word_probabilities(prompt)
            D_m = llm_model.get_choice_probs(word_probs, len(new_options))
            
            assert len(list(D_m.values())) == len(new_options)
            if config.data.dataset == 'opinion_qa':
                # Calculate Wasserstein distance
                if ph.get_max_wd(q_ordinal) == 0:
                    continue
                else:
                    wd = wasserstein_distance(q_ordinal, q_ordinal, list(D_m.values()), new_selections) / ph.get_max_wd(q_ordinal)
                    rep = 1 - wd
                    rep_values.append(rep)
            elif config.data.dataset == 'anthropic_global_opinions':
                # Calculate Jensen-Shannon divergence
                jd = jensenshannon(new_selections, list(D_m.values()))
                rep = 1 - jd
                rep_values.append(rep)
        rep_values_per_group[group_name] = sum(rep_values)/len(rep_values)
        if config.use_context:
            csv_filename = f"{dir_path}/few_shot{config.eval_n_ctx_qs}_{config.prompt_format}_{config.data.dataset}_steer{config.steer}_results.csv"
        else:
            csv_filename = f"{dir_path}/few_shot{config.eval_n_ctx_qs}_{config.prompt_format}_{config.data.dataset}_results.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Group', 'Alignment Score'])
            for group, rep_value in rep_values_per_group.items():
                csvwriter.writerow([group, rep_value])
if __name__ == "__main__":
    main()

