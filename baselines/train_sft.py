import datetime
import os
from typing import Dict, List, Optional
from hydra.utils import instantiate, get_original_cwd

import numpy as np
from data.constants import COUNTRIES, ALPHABET
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import (
    TrainingArguments, 
    Trainer,
)
from transformers.trainer_utils import speed_metrics
import time
import pandas as pd
import wandb
import string
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import torch
import math
import hydra 
from omegaconf import DictConfig, OmegaConf

from data.anthropic_global_opinions import (
    AnthropicDataCollator_meta,
    AnthropicDataCollator_sft,
)
import data.helpers as ph
from data.utils import get_alpaca_prompt, get_options_str, get_llama2_prompt
from utils import (
    set_random_seed,
    prepare_ds,
    INT_TO_GROUP,
    GROUP_NAMES,
    prepare_model_tokenizer
)

def prepare_collator(tokenizer, config):
    if config.data.task == "sft_pergroup":
        collator = AnthropicDataCollator_sft(tokenizer=tokenizer, prompt_format=config.prompt_format)
    elif config.data.task == 'oqa_opinions_meta_SFT':
        collator = AnthropicDataCollator_meta(tokenizer=tokenizer, prompt_format=config.prompt_format)
    else:
        raise ValueError(f"Unknown task {config.data.task}")
    
    return collator


class GroupAlignmentTrainer(Trainer):
    def __init__(self, config, *args, **kwargs):
        self.exp_config = config
        super().__init__(*args, **kwargs)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        
        alignment_score = self.compute_alignment_score(self.eval_dataset)
        output.metrics['eval_alignment_model_vs_group'] = alignment_score
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def get_predictions(self, sentence):
        # Encode the sentence using the tokenizer and return the model predictions.
        max_len = 4096 if self.exp_config.prompt_format == "llama2" else 2048
        inputs = self.tokenizer.encode(sentence, return_tensors="pt", max_length=max_len, truncation=True)
        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = outputs[0]
        return predictions

    def get_next_word_probabilities(self, sentence, top_k=200):
        with torch.no_grad():
            # Get the model predictions for the sentence.
            predictions = self.get_predictions(sentence)
            
            # Get the next token candidates.
            next_token_candidates_tensor = predictions[0, -1, :]

            # Get the top k next token candidates.
            topk_candidates_indexes = torch.topk(
                next_token_candidates_tensor, top_k).indices.tolist()
            
            # Filter the token probabilities for the top k candidates.
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

    def compute_alignment_score(self, ds):
        ds_df = pd.DataFrame(ds)
        # Initialize an empty list to hold D_m values
        dm_values = []
        for i, r in ds_df.iterrows():
            # TODO: batchify this
            question = r['questions']
            options = r['options']
            ordinal = r['ordinal']
            prompt = f"Question: {question}\n"
            instruction = "Answer the following question by picking from ONE of the given options."
            input_text = "{question}\n\nOptions:{options}".format(
                question=question, options=get_options_str(options))
            if self.exp_config.prompt_format == 'alpaca':
                prompt = get_alpaca_prompt(instruction=instruction, input_text=input_text)
            elif self.exp_config.prompt_format == 'llama2':
                prompt = get_llama2_prompt(user_message=input_text, system_prompt=instruction)
            word_probs = self.get_next_word_probabilities(prompt) 
            num_choices = len(ordinal)
            D_m = self.get_choice_probs(word_probs, num_choices)
            dm_values.append(list(D_m.values()))
            assert len(list(D_m.values())) == len(r['selections'])
        ds_df['d_m'] = dm_values
        if self.exp_config.data.dataset == 'anthropic_global_opinions':
            # for anthropic global opinions, we use jensen shannon distance
            ds_df['Distance'] = ds_df.apply(lambda x: jensenshannon(x['d_m'], x['selections']), axis=1)
        elif self.exp_config.data.dataset == 'opinion_qa':
            # for opinion qa, we use wasserstein distance
            ds_df['Distance'] = ds_df.swifter.apply(lambda x: wasserstein_distance(x['ordinal'], 
                                                                            x['ordinal'],
                                                                            x['d_m'], x['selections']) / ph.get_max_wd(x['ordinal']), 
                                            axis=1)
        
        ds_df['Rep'] = 1 - ds_df['Distance']
        alignment_score = ds_df['Rep'].mean()
        return alignment_score


@hydra.main(config_path="configs", config_name="train_sft")
def main(config: DictConfig) -> None:
    config.prompt_format = 'alpaca' if 'alpaca' in config.model_ckpt else 'llama2' if 'llama' in config.model_ckpt else None
    group_str = INT_TO_GROUP[config.data.group_idx] if config.data.dataset == 'opinion_qa' else COUNTRIES[config.data.group_idx]
    config.expid += f"sft{config.prompt_format}{config.data.dataset}{group_str}_nosteertrain_numq{config.data.train_nq}seed_{config.seed}_lr{config.trainer.learning_rate}"
    set_random_seed(config.seed)
    wandb.init(project=config.project_name, 
               notes=OmegaConf.to_yaml(config),
               name=config.expid)
    wandb.config.update(OmegaConf.to_container(config, resolve=True))
    
    # Prepare dataset and model
    ds = prepare_ds(config)
    model, tokenizer = prepare_model_tokenizer(config)

    dt_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(config.trainer.output_dir, config.expid, dt_now)
    args = TrainingArguments(
        num_train_epochs=config.trainer.num_train_epochs,
        per_device_train_batch_size=config.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=config.trainer.per_device_eval_batch_size,
        learning_rate=config.trainer.learning_rate,
        weight_decay=config.trainer.weight_decay,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy='steps',
        save_steps=300,
        save_total_limit = 2,
        load_best_model_at_end=False,
        logging_strategy="epoch",
        remove_unused_columns=False,
        bf16=config.trainer.bf16,
    )

    trainer = GroupAlignmentTrainer(
        config=config,
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        data_collator=prepare_collator(tokenizer, config),
    )

    trainer.train()



if __name__ == "__main__":
    main()






