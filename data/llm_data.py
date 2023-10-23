import torch
from torch.distributions import MultivariateNormal, StudentT
from attrdict import AttrDict
import math
import pandas as pd
import numpy as np
import os
from data.constants import COUNTRIES, ALPHABET
from datasets import (
    load_dataset, 
    Dataset,
)
import ast
import random

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
__all__ = ['collate_fn_gpo', 'OqaGroupDataset_gpo', 'GlobalGroupDataset_gpo', 'collate_fn_gpo_global', 'collate_fn_gpo_global_padding']


label_set = ['Northeast', 'South', 'Female', 'Male', 'College graduate/some postgrad',
             'Less than high school', 'Atheist', 'Hindu', 'Jewish', 'Muslim',
             'Protestant', 'Democrat', 'Republican', '$100,000 or more',
             'Less than $30,000', 'Conservative', 'Liberal', 'Moderate', 'Asian',
             'Black', 'Hispanic', 'White']

# Create an integer mapping for the labels
label_to_int = {label: idx for idx, label in enumerate(label_set)}

class OqaGroupDataset_gpo(Dataset):
    def __init__(self, df, config=None, device='cuda', seed=41):
        self.device = device
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        # Group by 'group' and then by 'qkey'
        grouped_by_group = df.groupby('group')
        
        self.data = []
        for group_name, group_data in grouped_by_group:
            total_num_options = 0
            qkey_list = []
            for qkey, sub_group in group_data.groupby('qkey'):
                embedding = torch.tensor(sub_group['embedding'].tolist())
                prob_y = torch.tensor(sub_group['prob_y'].values, dtype=torch.float).unsqueeze(-1)
                total_num_options += len(embedding)
                qkey_list.append({
                    'q_emb': embedding,
                    'prob_ys': prob_y,
                    'qkey': qkey
                })
            self.data.append({
                'groups': group_name,
                'qkeys': qkey_list,
                'total_nqs': total_num_options,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        group_data = self.data[idx]

        return {
            'questions': group_data['qkeys'],
            'groups': group_data['groups'],
            'total_nqs': group_data['total_nqs'],
        }


class GlobalGroupDataset_gpo(Dataset):
    def __init__(self, emb_df, dataset_groups, config, device='cuda', seed=41, mode='eval'):
        self.device = device
        # map embedding to raw selections
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        dataset = load_dataset("Anthropic/llm_global_opinions")["train"]
        df = pd.DataFrame(dataset)
        df['qkey'] = df.index
        
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
            for country in COUNTRIES:
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
        # Explode 'selections' into new rows
        df_filtered['selections'] = df_filtered['selections'].apply(lambda x: [(k, v) for k, v in x.items()])
        df_filtered = df_filtered.explode('selections')

        # Split tuple into separate 'group' and 'selections' columns
        df_filtered[['group', 'prob_y']] = pd.DataFrame(df_filtered['selections'].tolist(), index=df_filtered.index)
        df_filtered = df_filtered[df_filtered['group'].isin(COUNTRIES)]
        df = df_filtered.rename(columns={'question': 'questions'})
        # filter out df that groups are in train_groups:
        df = df[df['group'].isin(dataset_groups)]
        # Group by 'group' and then by 'qkey'
        grouped_by_group = df.groupby('group')
        self.data = []
        for group_name, group_data in grouped_by_group:
            qkey_list = []
            for qkey, sub_group in group_data.groupby('qkey'):
                embedding = torch.tensor(emb_df[emb_df['qkey'] == qkey]['embedding'].tolist())
                assert len(sub_group['options'].iloc[0]) == embedding.shape[0], "The number of options and the number of embeddings do not match."
                prob_y = torch.tensor(np.stack(sub_group['prob_y'].values), dtype=torch.float)
                qkey_list.append({
                    'q_emb': embedding,
                    'prob_ys': prob_y,
                    'qkey': qkey,
                    'question': emb_df[emb_df['qkey'] == qkey]['prompt_answer'].iloc[0]
                })
            self.data.append({
                'groups': group_name,
                'qkeys': qkey_list,
            })
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        group_data = self.data[idx]

        return {
            'questions': group_data['qkeys'],
            'groups': group_data['groups'],
        }


def collate_fn_gpo(batch, max_ctx_num_points, min_ctx_num_points, max_tar_num_points, min_tar_num_points, device='cuda'):
    total_num_qs = len(batch[0]['questions'])
    # Ensure min_ctx_num_points <= max_ctx_num_points and min_tar_num_points <= max_tar_num_points
    assert min_ctx_num_points <= max_ctx_num_points and min_tar_num_points <= max_tar_num_points, "The range values are not properly defined."
    assert max_ctx_num_points < total_num_qs
    # Randomly select the number of context points from range [min_ctx_num_points, max_ctx_num_points]
    num_ctx = torch.randint(low=min_ctx_num_points, high=max_ctx_num_points + 1, size=(1,)).item()

    min_tar = min_tar_num_points
    max_tar = min(max_tar_num_points, total_num_qs - num_ctx)
    num_tar = torch.randint(low=min_tar, high=max_tar + 1, size=(1,)).item()
    assert num_ctx + num_tar <= max_ctx_num_points + max_tar_num_points, "The total number of points exceeded the maximum limit."
    # Data holders
    collated_batch = { 
        'x': [],
        'xc': [],
        'xt': [],
        'y': [],
        'yc': [],
        'yt': [],
        'tarqlen':[],
    }
    perm_indices = torch.randperm(total_num_qs)
    ctx_indices = perm_indices[:num_ctx]
    tar_indices = perm_indices[num_ctx:num_ctx+num_tar]
    for b in batch:
        ctx_qs = [b['questions'][i] for i in ctx_indices]
        tar_qs = [b['questions'][i] for i in tar_indices]
        ctx_pa = torch.cat([q['q_emb'] for q in ctx_qs], dim=0)
        ctx_prob_ys = torch.cat([q['prob_ys'] for q in ctx_qs], dim=0)
        tar_pa = torch.cat([q['q_emb'] for q in tar_qs], dim=0)
        tar_prob_ys = torch.cat([q['prob_ys'] for q in tar_qs], dim=0)
        tar_q_len = torch.tensor([len(q['prob_ys']) for q in tar_qs])

        collated_batch['x'].append(torch.cat([ctx_pa, tar_pa], dim=0))
        collated_batch['xc'].append(ctx_pa)
        collated_batch['xt'].append(tar_pa)
        collated_batch['y'].append(torch.cat([ctx_prob_ys, tar_prob_ys], dim=0))
        collated_batch['yc'].append(ctx_prob_ys)
        collated_batch['yt'].append(tar_prob_ys)
        collated_batch['tarqlen'].append(tar_q_len)

    for key in collated_batch:
        collated_batch[key] = torch.stack(collated_batch[key]).to(device)
    
    return collated_batch

def collate_fn_gpo_global(batch, max_ctx_num_points, min_ctx_num_points, max_tar_num_points, min_tar_num_points, device='cuda'):
    num_ctx = torch.randint(low=min_ctx_num_points, high=max_ctx_num_points + 1, size=(1,)).item()
    min_tar = min_tar_num_points
    max_tar = max_tar_num_points
    num_tar = torch.randint(low=min_tar, high=max_tar + 1, size=(1,)).item()
    assert num_ctx + num_tar <= max_ctx_num_points + max_tar_num_points, "The total number of points exceeded the maximum limit."
    # Data holders
    collated_batch = { 
        'x': [],
        'xc': [],
        'xt': [],
        'y': [],
        'yc': [],
        'yt': [],
        'tarqlen':[],
    }
    temp_ctx_pa = []
    temp_tar_pa = []
    temp_ctx_prob_ys = []
    temp_tar_prob_ys = []
    for b in batch:
        num_questions = len(b['questions'])
        perm_indices = torch.randperm(num_questions)
        ctx_indices = perm_indices[:num_ctx]
        tar_indices = perm_indices[num_ctx:num_ctx+num_tar]
        ctx_qs = [b['questions'][i] for i in ctx_indices]
        tar_qs = [b['questions'][i] for i in tar_indices]

        ctx_pa = torch.cat([q['q_emb'] for q in ctx_qs], dim=0)
        ctx_prob_ys = torch.cat([q['prob_ys'][0] for q in ctx_qs], dim=0)
        tar_pa = torch.cat([q['q_emb'] for q in tar_qs], dim=0)
        tar_prob_ys = torch.cat([q['prob_ys'][0] for q in tar_qs], dim=0)
        
        temp_ctx_pa.append(ctx_pa)
        temp_tar_pa.append(tar_pa)
        temp_ctx_prob_ys.append(ctx_prob_ys)
        temp_tar_prob_ys.append(tar_prob_ys)
        
    # Find the minimum size for trimming
    min_ctx_size = min([x.shape[0] for x in temp_ctx_pa])
    min_tar_size = min([x.shape[0] for x in temp_tar_pa])

    # Trim and fill collated_batch
    for ctx_pa, tar_pa, ctx_prob_ys, tar_prob_ys in zip(temp_ctx_pa, temp_tar_pa, temp_ctx_prob_ys, temp_tar_prob_ys):
        ctx_pa = ctx_pa[:min_ctx_size]
        tar_pa = tar_pa[:min_tar_size]
        ctx_prob_ys = ctx_prob_ys[:min_ctx_size]
        tar_prob_ys = tar_prob_ys[:min_tar_size]
        collated_batch['x'].append(torch.cat([ctx_pa, tar_pa], dim=0))
        collated_batch['xc'].append(ctx_pa)
        collated_batch['xt'].append(tar_pa)
        collated_batch['y'].append(torch.cat([ctx_prob_ys, tar_prob_ys], dim=0))
        collated_batch['yc'].append(ctx_prob_ys)
        collated_batch['yt'].append(tar_prob_ys)

    for key in collated_batch:
        collated_batch[key] = torch.stack(collated_batch[key]).to(device)
    collated_batch['y'] = collated_batch['y'].reshape(len(batch), -1, 1)
    collated_batch['yc'] = collated_batch['yc'].reshape(len(batch), -1, 1)
    collated_batch['yt'] = collated_batch['yt'].reshape(len(batch), -1, 1)
    return collated_batch

def collate_fn_gpo_global_padding(batch, max_ctx_num_points, min_ctx_num_points, max_tar_num_points, min_tar_num_points, device='cuda'):
    num_ctx = torch.randint(low=min_ctx_num_points, high=max_ctx_num_points + 1, size=(1,)).item()
    min_tar = min_tar_num_points
    max_tar = max_tar_num_points
    num_tar = torch.randint(low=min_tar, high=max_tar + 1, size=(1,)).item()
    assert num_ctx + num_tar <= max_ctx_num_points + max_tar_num_points, "The total number of points exceeded the maximum limit."

    # Data holders
    collated_batch = { 
        'x': [],
        'xc': [],
        'xt': [],
        'y': [],
        'yc': [],
        'yt': [],
        'tarqlen': [], 
    }
    temp_ctx_pa = []
    temp_tar_pa = []
    temp_ctx_prob_ys = []
    temp_tar_prob_ys = []
    temp_tarqlen = []
    temp_ctxqlen = []

    for b in batch:
        num_questions = len(b['questions'])
        perm_indices = torch.randperm(num_questions)
        ctx_indices = perm_indices[:num_ctx]
        tar_indices = perm_indices[num_ctx:num_ctx+num_tar]
        ctx_qs = [b['questions'][i] for i in ctx_indices]
        tar_qs = [b['questions'][i] for i in tar_indices]

        ctx_pa = torch.cat([q['q_emb'] for q in ctx_qs], dim=0)
        ctx_prob_ys = torch.cat([q['prob_ys'][0] for q in ctx_qs], dim=0)
        tar_pa = torch.cat([q['q_emb'] for q in tar_qs], dim=0)
        tar_prob_ys = torch.cat([q['prob_ys'][0] for q in tar_qs], dim=0)
        
        temp_ctx_pa.append(ctx_pa)
        temp_tar_pa.append(tar_pa)
        temp_ctx_prob_ys.append(ctx_prob_ys)
        temp_tar_prob_ys.append(tar_prob_ys)
        temp_tarqlen.append(torch.tensor([len(q['prob_ys'][0]) for q in tar_qs]))
        temp_ctxqlen.append(torch.tensor([len(q['prob_ys'][0]) for q in ctx_qs]))

    pad_ctx_pa = pad_sequence(temp_ctx_pa, batch_first=True, padding_value=0)
    pad_tar_pa = pad_sequence(temp_tar_pa, batch_first=True, padding_value=0)
    pad_ctx_prob_ys = pad_sequence(temp_ctx_prob_ys, batch_first=True, padding_value=0)
    pad_tar_prob_ys = pad_sequence(temp_tar_prob_ys, batch_first=True, padding_value=0)

    # Fill collated_batch (Now no need to trim)
    for ctx_pa, tar_pa, ctx_prob_ys, tar_prob_ys in zip(pad_ctx_pa, pad_tar_pa, pad_ctx_prob_ys, pad_tar_prob_ys):
        collated_batch['x'].append(torch.cat([ctx_pa, tar_pa], dim=0))
        collated_batch['xc'].append(ctx_pa)
        collated_batch['xt'].append(tar_pa)
        collated_batch['y'].append(torch.cat([ctx_prob_ys, tar_prob_ys], dim=0))
        collated_batch['yc'].append(ctx_prob_ys)
        collated_batch['yt'].append(tar_prob_ys)
    collated_batch['tarqlen'] = temp_tarqlen

    for key in collated_batch:
        collated_batch[key] = torch.stack(collated_batch[key]).to(device)
    collated_batch['y'] = collated_batch['y'].reshape(len(batch), -1, 1)
    collated_batch['yc'] = collated_batch['yc'].reshape(len(batch), -1, 1)
    collated_batch['yt'] = collated_batch['yt'].reshape(len(batch), -1, 1)
    return collated_batch

