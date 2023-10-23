import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from attrdict import AttrDict
from torch.nn.functional import softmax
from torch.nn.utils.rnn import pad_sequence
from models.tnp import TNP


class GPO(TNP):
    def __init__(
        self,
        dim_x,
        dim_y,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers,
        bound_std=False
    ):
        super(GPO, self).__init__(
            dim_x,
            dim_y,
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            dropout,
            num_layers,
            bound_std
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y*2)
        )

    def forward(self, batch, reduce_ll=True):
        batch_size = batch['xc'].shape[0]
        target_real_lens = (torch.sum(batch['xt'], dim=-1) != 0).sum(1)
        assert torch.max(target_real_lens) == batch['yt'].shape[1], "Max target real lens is not equal to the number of target points"
        z_target = self.encode(batch, autoreg=False)
        out = self.predictor(z_target)
        mean, std = torch.chunk(out, 2, dim=-1)
        tar_q_len = batch['tarqlen']
        start_idx = 0
        softmax_mean = torch.zeros_like(mean)
        for bidx in range(batch_size):
            start_idx = 0
            for num_options in tar_q_len[bidx]: 
                    segment = mean[bidx, start_idx:start_idx + num_options]
                    softmax_segment = softmax(segment, dim=0)
                    softmax_mean[bidx, start_idx:start_idx + num_options] = softmax_segment  # Update the corresponding segment
                    start_idx += num_options
        mean = softmax_mean
        if self.bound_std:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            std = torch.exp(std)

        pred_tar = Normal(mean, std)
        log_probs = pred_tar.log_prob(batch['yt'])
        masked_log_probs = torch.zeros_like(log_probs)
        # Mask the log probabilities
        for i, length in enumerate(target_real_lens):
            masked_log_probs[i, :length] = log_probs[i, :length]

        outs = AttrDict()
        if reduce_ll:
            outs.tar_ll = masked_log_probs.sum(-1).mean()
        else:
            outs.tar_ll = masked_log_probs.sum(-1)
 
        outs.loss = - (outs.tar_ll)

        return outs

    def predict(self, xc, yc, xt):
        batch = AttrDict()
        batch.xc = xc
        batch.yc = yc
        batch.xt = xt
        batch.yt = torch.zeros((xt.shape[0], xt.shape[1], yc.shape[2]), device='cuda')

        z_target = self.encode(batch, autoreg=False)
        out = self.predictor(z_target)
        mean, std = torch.chunk(out, 2, dim=-1)
        if self.bound_std:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            std = torch.exp(std)

        return Normal(mean, std)