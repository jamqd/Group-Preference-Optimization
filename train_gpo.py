import os
import os.path as osp
import argparse
import yaml
import ast
import torch
import numpy as np
import time
import random
from torch.distributions import Normal
from scipy.stats import wasserstein_distance
from data.constants import COUNTRIES
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import uncertainty_toolbox as uct
from attrdict import AttrDict
from tqdm import tqdm
from copy import deepcopy
from data.llm_data import *
import wandb
import torch.multiprocessing as mp
import torch.nn.functional as F
from utils.misc import load_module
from utils.paths import results_path, evalsets_path
from utils.log import get_logger, RunningAverage, running_average_to_dict

from scipy.stats import entropy
from scipy.spatial import distance


def softmax_normalize(tensor):
    """Applies softmax normalization along the last dimension of the tensor"""
    return F.softmax(tensor, dim=-1)

def get_max_wd(ordered_ref_weights):
    d0, d1 = np.zeros(len(ordered_ref_weights)), np.zeros(len(ordered_ref_weights))
    d0[np.argmax(ordered_ref_weights)] = 1
    d1[np.argmin(ordered_ref_weights)] = 1
    max_wd = wasserstein_distance(ordered_ref_weights, ordered_ref_weights, d0, d1)
    return max_wd

class CollateFunction:
    def __init__(self, max_ctx_num_points, min_ctx_num_points, max_tar_num_points, min_tar_num_points, dataset='oqa'):
        self.max_ctx_num_points = max_ctx_num_points
        self.min_ctx_num_points = min_ctx_num_points
        self.max_tar_num_points = max_tar_num_points
        self.min_tar_num_points = min_tar_num_points
        self.dataset = dataset
    def __call__(self, batch):
        if self.dataset == 'oqa':
            return collate_fn_gpo(batch, self.max_ctx_num_points, self.min_ctx_num_points, self.max_tar_num_points, self.min_tar_num_points)
        elif self.dataset == 'globalqa':
            return collate_fn_gpo_global_padding(batch, self.max_ctx_num_points, self.min_ctx_num_points, self.max_tar_num_points, self.min_tar_num_points)

def main():
    mp.set_start_method('spawn') 
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--expid', type=str, default='')

    # Data
    parser.add_argument('--max_ctx_num_qs', type=int, default=100)
    parser.add_argument('--min_ctx_num_qs', type=int, default=10)
    parser.add_argument('--max_tar_num_qs', type=int, default=100)
    parser.add_argument('--min_tar_num_qs', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='globalqa', help='oqa or globalqa')
    parser.add_argument('--emb_model', type=str, default='alpaca')
    parser.add_argument('--exp_setup', type=str, default='gpo_trainsplit', help='Store meta train group splits')

    # Model
    parser.add_argument('--model', type=str, default="gpo")
    parser.add_argument('--emb', type=str, default="avg")
    parser.add_argument('--autoreg', type=bool, default=False)

    # Train
    parser.add_argument('--train_seed', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--train_num_samples', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--group_split', type=float, default=0.4)
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--eval_freq', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=1000)

    # Eval
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_qs', type=int, default=20)
    parser.add_argument('--eval_num_steps', type=int, default=10)
    parser.add_argument('--eval_logfile', type=str, default=None)
    parser.add_argument('--root', type=str, default=None)
    args = parser.parse_args()  

    if args.mode == 'eval':
        args.expid = 'eval'
    else:
        args.expid = args.expid + 'split' + str(args.group_split) + '_seed' + str(args.train_seed) + '_' + str(args.model) + str(args.dataset) + '_emb' + str(args.emb_model) + '_lr' + str(args.lr) + 'evalnq_' + str(args.eval_num_qs) 
    args.eval_setup = os.path.join(results_path, args.exp_setup)
    if args.root is None:
        if args.expid is not None:
            args.root = osp.join(results_path, 'group_alignment', args.model, args.expid)
        else:
            args.root = osp.join(results_path, 'group_alignment', args.model)

    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    with open(f'configs/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)

    ## LOAD EMBEDDINGS ##
    wordir = './baselines/get_emb/' ##TODO set to your own directory
    args.pickle_file_path = wordir + f'embeddings_{args.emb_model}_{args.dataset}.pkl'
    if 'alpaca' in args.pickle_file_path:
        config['dim_x'] = 4096
    if 'llama' in args.pickle_file_path:
        config['dim_x'] = 5120

    model = model_cls(**config)
    model.cuda()

    if args.dataset == 'oqa':
        wandb.init(project='group-alignment-gpo-oqa', name=args.expid, config=args)
    elif args.dataset == 'globalqa':
        wandb.init(project='group-alignment-gpo-anthropic', name=args.expid, config=args)
    wandb.config.update(config)

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        eval(args, model)
    

def load_datasets(args):
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
    random.seed(args.train_seed)

    df = pd.read_pickle(args.pickle_file_path)
    # Split DataFrame into a training set and an evaluation set by group.
    if args.dataset == 'oqa':
        groups = df['group'].unique()
        train_groups = np.random.choice(groups, size=int(len(groups)*args.group_split), replace=False)
        eval_groups = [group for group in groups if group not in train_groups]
    elif args.dataset == 'globalqa':
        groups = COUNTRIES
        train_groups = np.random.choice(groups, size=int(len(groups)*args.group_split), replace=False)
        eval_groups = [group for group in groups if group not in train_groups]
    if not os.path.exists(args.exp_setup):
        os.mkdir(args.exp_setup)
    with open(f"{args.exp_setup}/{args.expid}_eval_groups.txt", "w") as f:
        for group in eval_groups:
            f.write(f"{group}\n")
    print(eval_groups,'eval groups')
    print(train_groups,'train groups')

    if args.dataset == 'oqa':
        train_mask = df['group'].isin(train_groups)
        eval_mask =  df['group'].isin(eval_groups)
        train_df = df[train_mask]
        eval_df = df[eval_mask]
        train_dataset = OqaGroupDataset_gpo(train_df, config=args)
        eval_dataset = OqaGroupDataset_gpo(eval_df, config=args)
        return train_df, eval_df, train_dataset, eval_dataset
    elif args.dataset == 'globalqa':
        train_dataset = GlobalGroupDataset_gpo(df, train_groups, config=args, mode='train')
        eval_dataset = GlobalGroupDataset_gpo(df, eval_groups, config=args, mode='eval')
        return None, None, train_dataset, eval_dataset
    

def train(args, model):
    torch.set_num_threads(2)
    if osp.exists(args.root + '/ckpt.tar'):
        if args.resume is None:
            raise FileExistsError(args.root)
    else:
        os.makedirs(args.root, exist_ok=True)

    with open(osp.join(args.root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)
    
    train_df, eval_df, train_dataset, eval_dataset = load_datasets(args)
    collate_function = CollateFunction(args.max_ctx_num_qs, args.min_ctx_num_qs, args.max_tar_num_qs, args.min_tar_num_qs, dataset=args.dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=collate_function, num_workers=0)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, collate_fn=collate_function, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_steps)


    ravg = RunningAverage()
    start_step = 1
    best_alignscore = 0
    assert next(model.parameters()).is_cuda

    for step in tqdm(range(start_step, args.num_steps+1)):
        model.train()
        optimizer.zero_grad()
        if step == 1:
            if args.dataset == 'oqa':
                calculate_WD(args, model, eval_df, mode='eval')
            elif args.dataset == 'globalqa':
                calculate_JD(args, model, eval_dataset, mode='eval')

        for batch in train_dataloader:
            batch = {k: v.to('cuda') for k, v in batch.items()}
            outs = model(batch)
            outs.loss.backward()
            optimizer.step()
            scheduler.step()

        for key, val in outs.items():
            ravg.update(key, val)

        if step % args.eval_freq == 0:
            line = f'{args.model}:{args.expid} step {step} '
            line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
            line += f"[train_loss] "
            line += ravg.info()
            wandb.log(running_average_to_dict(ravg))
            
            if args.dataset == 'oqa':
                eval_alignment_score = calculate_WD(args, model, eval_df, mode='eval')
            elif args.dataset == 'globalqa':
                eval_alignment_score = calculate_JD(args, model, eval_dataset, mode='eval')
            if eval_alignment_score > best_alignscore:
                best_alignscore = eval_alignment_score
                ckpt = AttrDict()
                ckpt.model = model.state_dict()
                ckpt.optimizer = optimizer.state_dict()
                ckpt.scheduler = scheduler.state_dict()
                ckpt.step = step + 1
                
            if step % (5 * args.eval_freq) == 0:
                if args.dataset == 'oqa':
                    calculate_WD(args, model, train_df, mode='train')
                elif args.dataset == 'global':
                    calculate_JD(args, model, train_dataset, mode='train')
                torch.save(ckpt, os.path.join(args.root, f'ckpt_{step}.tar'))
                print('saved model to ',args.root)

            ravg.reset()

    args.mode = 'eval'
    eval(args, model)
    wandb.finish()


def eval(args, model):
    if args.mode == 'eval':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'), map_location='cuda')
        model.load_state_dict(ckpt.model)
    
    ravg = RunningAverage()
    model.eval()
    train_df, eval_df, train_dataset, eval_dataset = load_datasets(args)
    print('evaluating for dataset:', args.dataset, 'emb model:', args.emb_model, 'eval_num_qs:', args.eval_num_qs, 'group_split:', args.group_split)
    if args.dataset == 'oqa':
        calculate_WD(args, model, eval_df, mode='eval', logging=False)
    elif args.dataset == 'globalqa':
        calculate_JD(args, model, eval_dataset, mode='eval', logging=False)
    return 

def calculate_JD(args, model, dataset, mode='eval', logging=True):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    distances_all = []
    for i, batch in enumerate(dataloader):
        distances_group = []
        this_group = batch['groups']
        group_questions = batch['questions']
        num_questions = len(group_questions)
        context_questions = np.random.choice(np.arange(num_questions), size=args.eval_num_qs, replace=False)
        target_questions = np.setdiff1d(np.arange(num_questions), context_questions)
        # Now, let's collect the context embeddings and probabilities.
        ctx_embeddings = []
        ctx_prob_ys = []
        tar_embeddings = []
        tar_prob_ys = []
        for context_q_idx in context_questions:
            ctx_embeddings.append(group_questions[context_q_idx]['q_emb'])
            ctx_prob_ys.append(group_questions[context_q_idx]['prob_ys'][0])
        ctx_embeddings = torch.cat(ctx_embeddings, dim=1).to('cuda')
        ctx_prob_ys = torch.cat(ctx_prob_ys, dim=1).unsqueeze(-1).to('cuda', dtype=torch.float)
        for target_q_idx in target_questions:
            tar_embeddings = group_questions[target_q_idx]['q_emb'].to('cuda')
            tar_prob_ys = group_questions[target_q_idx]['prob_ys'].to('cuda')
            with torch.no_grad():
                predicted_distribution = model.predict(ctx_embeddings, ctx_prob_ys, tar_embeddings).loc
                predicted_distribution = softmax_normalize(predicted_distribution.reshape(-1))
                    

                D_H = tar_prob_ys
                D_H_np = np.array(D_H.cpu())
                D_H_np = D_H_np.squeeze()
                predicted_distribution_np = predicted_distribution.cpu().detach().numpy().squeeze()
                normalized_jd = distance.jensenshannon(predicted_distribution_np, D_H_np)
                if torch.isnan(torch.tensor(normalized_jd)).any():
                    normalized_jd = 0.0
                distances_all.append(normalized_jd)
                distances_group.append(normalized_jd)
        mean_distance_group = np.mean(distances_group)
        if logging:
            wandb.log({f"{mode.capitalize()}_alignment_score_{this_group}": 1 - mean_distance_group})
        print(f"{mode.capitalize()}_alignment_score_{this_group}:  {1 - mean_distance_group}")
    mean_distance = np.mean(distances_all)
    print(f"{mode.capitalize()} Mean Jensen Divergence: {mean_distance} Mean alignment score:{1-mean_distance}")
    if logging:
        wandb.log({f"{mode.capitalize()}_alignment_score_mean_testgroup": 1-mean_distance})
    return 1-mean_distance

def calculate_WD(args, model, df, mode='eval', logging=True):
    model.eval()
    groups = df['group'].unique()
    unique_questions = df['qkey'].unique()
    context_questions = np.random.choice(unique_questions, size=args.eval_num_qs, replace=False)
    target_questions = np.setdiff1d(unique_questions, context_questions)
    distances_all = []

    for grp_idx, group in enumerate(groups):
        group_df = df[df['group'] == group]
        distances_group = []
        for idx, question in enumerate(target_questions):
            # Get the dataframe for the current question
            question_df = group_df[group_df['qkey'] == question]
            # Extract embeddings and probabilities
            embeddings = torch.stack([torch.tensor(e) for e in question_df['embedding'].tolist()]).unsqueeze(0).to('cuda')
            # Get the context for the current question
            context_df = group_df[group_df['qkey'].isin(context_questions)]
            context_embeddings = torch.stack([torch.tensor(e) for e in context_df['embedding'].tolist()]).unsqueeze(0).to('cuda')
            context_prob_ys = torch.tensor(context_df['prob_y'].values, dtype=torch.float).unsqueeze(0).unsqueeze(-1).to('cuda')
            if torch.isnan(context_embeddings).any():
                print("Warning: NaN values detected in context_embeddings!")
            with torch.no_grad():
                predicted_distribution_list = []  # Renamed to make it clearer that this is a list
                for i, single_embedding in enumerate(embeddings.squeeze(0)):
                    single_embedding = single_embedding.unsqueeze(0).unsqueeze(0)  # Add the batch and sequence dimensions back
                    # Generate prediction for the current embedding
                    single_predicted_distribution = model.predict(context_embeddings, context_prob_ys, single_embedding)
                    # Normalize the single predicted distribution
                    single_predicted_distribution = single_predicted_distribution.loc  # Take mean over sample dimension if needed
                    predicted_distribution_list.append(single_predicted_distribution)
                predicted_distribution = torch.stack(predicted_distribution_list)
                predicted_distribution = softmax_normalize(predicted_distribution.reshape(-1))
            
            # Convert the string representation of the list to an actual list
            D_H = ast.literal_eval(question_df['D_H'].iloc[0])

            # Convert the list to a tensor
            D_H = torch.tensor(D_H, dtype=torch.float).to('cuda')
            # Convert predicted_distribution and D_H to numpy
            predicted_distribution_np = predicted_distribution.cpu().detach().numpy()
            predicted_distribution_np = np.squeeze(predicted_distribution_np)
            
            D_H_np = np.array(D_H.cpu())
            ordinal = ast.literal_eval(question_df['ordinal'].iloc[0])
            ordinal_np = np.array(ordinal, dtype=float)
            # Compute Wasserstein distance
            if get_max_wd(ordinal_np) == 0:
                continue
            else:
                epsilon = 0
                normalized_wd = wasserstein_distance(ordinal_np, ordinal_np, predicted_distribution_np, D_H_np) / (get_max_wd(ordinal_np) + epsilon)
                distances_group.append(normalized_wd)
                distances_all.append(normalized_wd)

        mean_distance_group = np.mean(distances_group)
        if logging:
            wandb.log({f"{mode.capitalize()}_alignment_score_{group}": 1 - mean_distance_group})
        print(f"{mode.capitalize()}_alignment_score_{group}:  {1 - mean_distance_group}")
        
    # Compute the mean Wasserstein distance
    mean_distance = np.mean(distances_all)
    print(f"{mode.capitalize()} Mean Wasserstein Distance: {mean_distance} Mean alignment score:{1-mean_distance}")
    if logging:
        wandb.log({f"{mode.capitalize()}_alignment_score_mean_testgroup": 1-mean_distance})
    return 1 - mean_distance
    
if __name__ == '__main__':

    main()
