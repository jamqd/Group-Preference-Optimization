import datetime
import os
from typing import Dict, List, Optional
import numpy as np
import random
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments, 
    Trainer
)
from data.constants import COUNTRIES, ALPHABET
from transformers.trainer_utils import speed_metrics
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
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
    AnthropicDataCollator_sft,
    AnthropicDataCollator_meta,
    process_example_meta
)
import data.helpers as ph
from utils import (
    set_random_seed,
    prepare_model_tokenizer,
    prepare_ds
)

import warnings
warnings.filterwarnings("ignore")
torch.set_num_threads(1)

def prepare_collator(tokenizer, config):
    if config.data.task == "sft_pergroup":
        collator = AnthropicDataCollator_sft(tokenizer=tokenizer)
    elif config.data.task == 'meta_SFT':
        collator = AnthropicDataCollator_meta(tokenizer=tokenizer, num_meta_questions=config.data.train_nq, prompt_format=config.prompt_format)
    else:
        raise ValueError(f"Unknown task {config.data.task}")
    
    return collator
    

class GroupAlignmentTrainer(Trainer):
    def __init__(self, config, eval_n_ctx_qs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_n_ctx_qs = eval_n_ctx_qs
        self.exp_config = config

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
            if token in choices and len(choice_probs[token]) < 1:
                choice_probs[token].append(np.exp(logit))        
        # Compute mean log probabilities or set to -inf if list is empty
        for key, value_list in choice_probs.items():
            choice_probs[key] = np.mean(value_list) if value_list else min_logit

        # Normalize
        total = sum(choice_probs.values())
        for c in choice_probs:
            choice_probs[c] /= total

        return choice_probs

    def compute_alignment_score(self, ds):
        rep_values = []
        with torch.no_grad():
            for ex in ds:
                group_df = ex
                random_indices = group_df.sample(self.eval_n_ctx_qs).index.tolist() 
                context_indices = random_indices
                meta_questions = group_df.loc[context_indices, 'questions'].tolist()
                meta_selections = group_df.loc[context_indices, 'selections'].tolist()
                meta_options = group_df.loc[context_indices, 'options'].tolist()
                remaining_indices = group_df.index.drop(context_indices).tolist()
                sample_size = int(len(remaining_indices) * 0.2)
                random_sample_indices = random.sample(remaining_indices, sample_size)
                # NOTE: Here we only loop over the randomly chosen 20% indices for faster eval on the fly because it takes much time.
                for target_q in random_sample_indices:
                    # Now use main_index for the new question
                    new_question = group_df.loc[target_q, 'questions']
                    new_selections = group_df.loc[target_q, 'selections']
                    new_options = group_df.loc[target_q, 'options']
                    q_ordinal = group_df.loc[target_q, 'ordinal']
                    prompt = process_example_meta(
                        meta_questions, meta_selections, meta_options,
                        new_question, new_selections, new_options, prompt_format=self.exp_config.prompt_format
                    )
                    word_probs = self.get_next_word_probabilities(prompt) 
                    num_choices = len(new_options)
                    D_m = self.get_choice_probs(word_probs, num_choices)
                    assert len(list(D_m.values())) == len(new_options)
                    if self.exp_config.data.dataset == 'opinion_qa':
                        # Calculate Wasserstein distance
                        if ph.get_max_wd(q_ordinal) == 0:
                            continue
                        else:
                            wd = wasserstein_distance(q_ordinal, q_ordinal, list(D_m.values()), new_selections) / ph.get_max_wd(q_ordinal)
                            rep = 1 - wd
                            rep_values.append(rep)
                    elif self.exp_config.data.dataset == 'anthropic_global_opinions':
                        # Calculate Jensen-Shannon divergence
                        jd = jensenshannon(new_selections, list(D_m.values()))
                        rep = 1 - jd
                        rep_values.append(rep)
        # Calculate the mean of the reputation metric
        alignment_score = sum(rep_values) / len(rep_values)
        return alignment_score


@hydra.main(config_path="configs", config_name="train_incontextft")
def main(config: DictConfig) -> None:
    """Set up parameters based on the configuration and initialize wandb."""
    config.prompt_format = 'alpaca' if 'alpaca' in config.model_ckpt else 'llama2' if 'llama' in config.model_ckpt else None
    config.expid += f"lr{config.trainer.learning_rate}meta_{config.prompt_format}_split{config.data.group_split}ctx_numq{config.data.train_nq}seed{config.seed}{config.data.dataset}"
    set_random_seed(config.seed)
    dataset_name_map = {
        'anthropic_global_opinions': 'group-alignment-sft-anthropic-ictxft',
        'opinion_qa': 'group-alignment-sft-oqa-ictxft'
    }
    config.project_name = dataset_name_map.get(config.data.dataset)
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
        eval_steps=200,
        save_strategy='steps',
        save_steps=400,
        save_total_limit = 2,
        load_best_model_at_end=True,
        logging_strategy="epoch",
        remove_unused_columns=False,
        resume_from_checkpoint = True,
        bf16=config.trainer.bf16,
    )

    trainer = GroupAlignmentTrainer(
        model=model,
        config=config,
        args=args,
        eval_n_ctx_qs=config.data.train_nq,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        data_collator=prepare_collator(tokenizer, config),
    )
    # Get the maximum accepted token size for the model
    max_token_size = model.config.max_position_embeddings
    print(f"The model's maximum accepted token size is: {max_token_size}")
    
    trainer.train()
    
if __name__ == "__main__":
    main()

