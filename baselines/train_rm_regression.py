import datetime
import os
from typing import Dict, List, Optional
from hydra.utils import instantiate, get_original_cwd
import ray
from ray.data.preprocessors import BatchMapper
from ray.data.preprocessors import Chain
from ray import train
from tqdm import tqdm
from ray.air import session, Checkpoint
from ray.train.huggingface import TransformersTrainer
from utils import (
    set_random_seed,
    prepare_model_tokenizer,
    prepare_ds
)
import evaluate
import numpy as np
import torch.nn as nn
import random
from data.constants import COUNTRIES, ALPHABET
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    default_data_collator,
)
import argparse
from transformers.trainer_utils import speed_metrics
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
import time
import pandas as pd
import wandb
import string
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import torch
import swifter
import json
import math
import hydra 
from omegaconf import DictConfig, OmegaConf

from data.anthropic_global_opinions import (
    get_dataset_Global, 
    get_dataset_oqa,
    AnthropicDataCollator_meta,
    AnthropicDataCollator_sft,
    collator_regress_rm,
)
import data.helpers as ph
from data.utils import get_alpaca_prompt, get_options_str, get_llama2_prompt
from utils import (
    print_trainable_parameters, 
    set_random_seed,
    prepare_ds,
    set_available_gpus,
    INT_TO_GROUP,
    GROUP_NAMES
)

def prepare_collator(tokenizer, config):
    if config.data.task == "sft_pergroup":
        collator = AnthropicDataCollator_sft(tokenizer=tokenizer, prompt_format=config.prompt_format)
    elif config.data.task == 'oqa_opinions_meta_SFT':
        collator = AnthropicDataCollator_meta(tokenizer=tokenizer, prompt_format=config.prompt_format)
    elif config.data.task == 'reward_model_regress':
        collator = collator_regress_rm(tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown task {config.data.task}")
    
    return collator


class GroupAlignmentTrainer(Trainer):
    def __init__(self, config, *args, **kwargs):
        self.exp_config = config
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")  # Assuming that your model returns logits

        # Convert labels to a PyTorch tensor and move it to the same device as logits
        if isinstance(labels, list):
            labels = torch.tensor(labels).to(logits.device).float()
        else:
            labels = labels.to(logits.device).float()

        self.act = nn.Sigmoid()
        # Make sure to create a new variable instead of modifying logits in-place
        activated_logits = self.act(logits)

        # Compute MSE Loss
        loss_fct = nn.MSELoss()

        # # If logits and labels have additional dimensions you may want to adjust the view.
        total = torch.sum(torch.exp(activated_logits.view(-1)))
        normalized_logits = activated_logits / total
        loss = loss_fct(normalized_logits.view(-1), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

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
        alignment_score = self.compute_alignment_score(self.eval_dataset)
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        
        
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
        inputs = self.tokenizer.encode(sentence, return_tensors="pt", max_length=2048)
        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = outputs[0]
        return predictions

    def get_next_word_probabilities(self, sentence, top_k=500):

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
        choice_probs = {}
        for c in choices:
            choice_probs[c] = []
        for token, prob in raw_token_probs:
            if token in choices:
                choice_probs[token].append(prob)
        for key, value in choice_probs.items():
            choice_probs[key] = np.mean(value)
        # Exponentiate each value
        exp_choice_probs = {c: math.exp(v) for c, v in choice_probs.items()}
        total = sum(exp_choice_probs.values())
        # Normalize
        for c in exp_choice_probs:
            exp_choice_probs[c] /= total
        return exp_choice_probs

    def compute_alignment_score(self, ds):
        ds_df = pd.DataFrame(ds)
        # Initialize an empty list to hold D_m values
        batch_size = 4  # Set your batch size
        dm_values = []

        # Iterate over the DataFrame in batches
        for i in tqdm(range(0, len(ds_df), batch_size)):
            batch = ds_df.iloc[i:i + batch_size]
            batch_prompts = []

            # Collecting all the prompts for the current batch
            for _, r in batch.iterrows():
                question = r['questions']
                options = r['options']
                ordinal = r['ordinal']

                prompt = f"Question: {question}\n"
                instruction = "Answer the following question by picking from the given options"
                input_text = "{question}\n\nOptions:{options}".format(question=question, options=get_options_str(options))

                if self.exp_config.prompt_format == 'alpaca':
                    prompt = get_alpaca_prompt(instruction=instruction, input_text=input_text)
                elif self.exp_config.prompt_format == 'llama2':
                    prompt = get_llama2_prompt(user_message=input_text, system_prompt=instruction)

                for idx in range(len(options)):
                    this_prompt = prompt + ALPHABET[idx]
                    batch_prompts.append(this_prompt)

            # Tokenize all the prompts in one go
            inputs = self.tokenizer(batch_prompts, padding=True, return_tensors="pt", max_length=4096)

            # Run the model once for the entire batch
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.get("logits")
            self.act = nn.Sigmoid()
            logits = torch.exp(self.act(logits))

            # Split logits to individual rows and normalize
            start_idx = 0
            for _, r in batch.iterrows():
                end_idx = start_idx + len(r['options'])
                D_m = logits[start_idx:end_idx]
                total = torch.sum(D_m)
                D_m /= total
                dm_values.append(D_m.numpy().flatten().tolist())
                assert len(D_m) == len(r['selections'])
                start_idx = end_idx

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


@hydra.main(config_path="configs", config_name="train_regress_rm")
def main(config: DictConfig) -> None:
    group_str = INT_TO_GROUP[config.data.group_idx] if config.data.dataset == 'opinion_qa' else COUNTRIES[config.data.group_idx]
    if 'alpaca' in config.model_ckpt:
        config.prompt_format = 'alpaca'
        config.data.train_nq = 15
    elif 'lama2' in config.model_ckpt:
        config.prompt_format = 'llama2'
        config.data.train_nq = 20
    config.expid = f"regressrm_{config.prompt_format}_{config.data.dataset}_{group_str}_numq{config.data.train_nq}_seed{config.seed}"

    set_random_seed(config.seed)
    wandb.init(project=config.project_name, 
               notes=OmegaConf.to_yaml(config),
               name=config.expid)
    wandb.config.update(OmegaConf.to_container(config, resolve=True))
    ds = prepare_ds(config)
    
    model, tokenizer = prepare_model_tokenizer(config, reward_model=True)

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
        eval_steps=150,
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
    torch.autograd.set_detect_anomaly(True)
    trainer.train()



if __name__ == "__main__":
    main()






