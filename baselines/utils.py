import random
import numpy as np
import torch
from typing import Union, List
import os
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
import pandas as pd
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, PeftModel, PeftConfig
from datasets import (
    load_dataset, 
    Dataset,
)
from data.anthropic_global_opinions import (
    get_dataset_Global, 
    get_dataset_Global_meta,
    get_dataset_oqa,
    get_dataset_oqa_meta,
)
def set_available_gpus(gpu_ids: Union[int, List[int]]):
    available_gpus = []
    if isinstance(gpu_ids, int):
        if gpu_ids == -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                [str(i) for i in range(torch.cuda.device_count())])
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            available_gpus = [gpu_ids]
    else: 
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpu_ids])
    return available_gpus

GROUP_NAMES = ['Northeast', 'South', 'Female', 'Male', 'College graduate/some postgrad',
             'Less than high school', 'Atheist', 'Hindu', 'Jewish', 'Muslim',
             'Protestant', 'Democrat', 'Republican', '$100,000 or more',
             'Less than $30,000', 'Conservative', 'Liberal', 'Moderate', 'Asian',
             'Black', 'Hispanic', 'White']

INT_TO_GROUP = {
    0: 'Northeast', 1: 'South', 2: 'Female', 3: 'Male', 4: 'College_graduatesome_postgrad',
    5: 'Lessthanhighschool', 6: 'Atheist', 7: 'Hindu', 8: 'Jewish', 9: 'Muslim',
    10: 'Protestant', 11: 'Democrat', 12: 'Republican', 13: '$100000ormore',
    14: 'Lessthan30000', 15: 'Conservative', 16: 'Liberal', 17: 'Moderate', 18: 'Asian',
    19: 'Black', 20: 'Hispanic', 21: 'White'
}


COUNTRIES = [
    'Nigeria', 
    'Egypt', 
    'India (Current national sample)', 
    'China', 
    'Japan', 
    'Germany', 
    'France', 
    'Spain', 
    'United States', 
    'Canada', 
    'Brazil', 
    'Argentina', 
    'Australia', 
    'New Zealand'
]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} \
            || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def prepare_model_tokenizer(config, reward_model=False, load_pretrained=False, load_path=None):
    print(f"Loading model from {config.model_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_ckpt)
    if reward_model:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_ckpt,
            load_in_8bit=config.use_int8,
            num_labels=1,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_ckpt, 
                                                 load_in_8bit=config.use_int8)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    if config.use_int8:
        model = prepare_model_for_int8_training(model)
        
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )
    if load_pretrained:
        print(load_path,'loading from here')
        model = PeftModel.from_pretrained(model, load_path, is_trainable=False)
    else:
        model = get_peft_model(model, lora_config)
    
    print_trainable_parameters(model)

    return model, tokenizer

def prepare_ds(config):
    if config.data.dataset == "anthropic_global_opinions":
        if config.data.task == "sft_pergroup" or config.data.task == 'reward_model_regress':
            ds = get_dataset_Global(config)
        elif config.data.task == 'meta_SFT':
            ds = get_dataset_Global_meta(config)
    elif config.data.dataset == "opinion_qa":
        if config.data.task == "sft_pergroup" or config.data.task == 'reward_model_regress':
            ds = get_dataset_oqa(config.data.group_idx,
                                config.data.oqa_datapath)
        elif config.data.task == 'meta_SFT':
            ds = get_dataset_oqa_meta(config.data.oqa_datapath)
    else:
        raise ValueError(f"Unknown dataset {config.data.dataset}")
    
    if config.data.task == 'sft_pergroup' or config.data.task == 'reward_model_regress':
        # Extract all qkeys
        all_qkeys = [item['qkey'] for item in ds]

        # Shuffle qkeys
        random.seed(config.seed)
        random.shuffle(all_qkeys)

        # Select n for training
        n = config.data.train_nq 
        train_qkeys = all_qkeys[:n]
        test_qkeys = all_qkeys[n:]
        
        print("Train dataset size:", len(train_qkeys))
        print("Test dataset size:", len(test_qkeys))
        
        # Filter datasets based on qkeys
        train_ds = [item for item in ds if item['qkey'] in train_qkeys]
        test_ds = [item for item in ds if item['qkey'] in test_qkeys]

        # Create a DataFrame
        df = pd.DataFrame({
            'train_qkeys': pd.Series(train_qkeys).astype(str),
            'test_qkeys': pd.Series(test_qkeys).astype(str)
        })

        # Save the train test questions to csv for fair coomparison
        df.to_csv(config.trainer.reproduce_exp_log_dir + config.expid + 'train_test_qkeys.csv', index=False)

        # Check for overlap (although there shouldn't be any if done correctly)
        overlap = set(train_qkeys).intersection(set(test_qkeys))
        if overlap:
            raise ValueError(f"Train and test qkeys should not overlap. Overlapping keys: {overlap}")
        return {'train': train_ds, 'test': test_ds}
    
    elif config.data.task == 'meta_SFT':
        df = pd.DataFrame(ds)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

        # Get all unique groups
        all_groups = set(df['group'].unique())

        # randomly sample test groups:
        train_groups = set(np.random.choice(list(all_groups), size=int(len(all_groups)*config.data.group_split), replace=False))
        test_groups = all_groups - set(train_groups)
        train_groups = list(train_groups)
        test_groups = list(test_groups)
        print(all_groups,'all groups')
        print(test_groups,'test groups')
        print(train_groups, 'train groups')

        # Save train and test groups to a CSV file, store for reproducibility
        train_test_groups_df = pd.DataFrame({
            'train_groups': pd.Series(train_groups).astype(str),
            'test_groups': pd.Series(test_groups).astype(str)
        })
        if config.data.task == 'meta_SFT':
            config.trainer.reproduce_exp_log_dir += 'meta_sft_logs' 
        elif config.data.task == 'sft_pergroup':
            config.trainer.reproduce_exp_log_dir += 'sft_logs'
        train_test_groups_df.to_csv(f'{config.trainer.reproduce_exp_log_dir}/{config.expid}_train_test_groups.csv', index=False)

        # Group the DataFrame by 'group' and create a list of DataFrames
        grouped = list(df.groupby('group'))
        list_of_dataframes_train = [group for name, group in grouped if name in train_groups]
        list_of_dataframes_test = [group for name, group in grouped if name in test_groups]
        
        return {'train': list_of_dataframes_train, 'test': list_of_dataframes_test}



