import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.irpe import get_rpe_config
from models.irpe import build_rpe
import math

class Attention(nn.Module):
    def __init__(self, cfg):
        super(Attention, self).__init__()
        if cfg.model.change_detector.att_dim % cfg.model.change_detector.att_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (cfg.model.change_detector.att_dim, cfg.model.change_detector.att_head))
        self.num_attention_heads = cfg.model.change_detector.att_head
        self.attention_head_size = int(cfg.model.change_detector.att_dim / cfg.model.change_detector.att_head)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(cfg.model.change_detector.att_dim, self.all_head_size)
        self.key = nn.Linear(cfg.model.change_detector.att_dim, self.all_head_size)
        self.value = nn.Linear(cfg.model.change_detector.att_dim, self.all_head_size)

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(cfg.model.change_detector.att_dim, eps=1e-6)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        # attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer += query_states
        context_layer = self.layer_norm(context_layer)
        return context_layer


class SelfAttention(nn.Module):
    def __init__(self, cfg, rpe_config=None):
        super(SelfAttention, self).__init__()
        if cfg.model.change_detector.att_dim % cfg.model.change_detector.att_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (cfg.model.change_detector.att_dim, cfg.model.change_detector.att_head))
        self.num_attention_heads = cfg.model.change_detector.att_head
        self.attention_head_size = int(cfg.model.change_detector.att_dim / cfg.model.change_detector.att_head)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(cfg.model.change_detector.att_dim, self.all_head_size)
        self.key = nn.Linear(cfg.model.change_detector.att_dim, self.all_head_size)
        self.value = nn.Linear(cfg.model.change_detector.att_dim, self.all_head_size)

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(cfg.model.change_detector.att_dim, eps=1e-6)

        # image relative position encoding
        self.rpe_q, self.rpe_k, self.rpe_v = \
            build_rpe(rpe_config,
                      head_dim=self.attention_head_size,
                      num_heads=self.num_attention_heads)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        # attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        if self.rpe_q is not None:
            attention_scores += self.rpe_q(query_layer)

        # image relative position on queries
        if self.rpe_k is not None:
            attention_scores += self.rpe_k(key_layer).transpose(2, 3)
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        # image relative position on values
        if self.rpe_v is not None:
            context_layer += self.rpe_v(attention_probs)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        context_layer += query_states
        context_layer = self.layer_norm(context_layer)
        return context_layer


class ChangeDetectorDoubleAttDyn(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.model.change_detector.input_dim
        self.dim = cfg.model.change_detector.dim
        self.feat_dim = cfg.model.change_detector.feat_dim
        self.att_head = cfg.model.change_detector.att_head
        self.att_dim = cfg.model.change_detector.att_dim

        # self.img = nn.Linear(self.feat_dim, self.att_dim)
        self.img = nn.Sequential(
            nn.Linear(self.feat_dim, self.att_dim),
            # nn.LayerNorm(self.att_dim, eps=1e-6),
            # nn.Dropout(0.1)
        )
        rpe_config = get_rpe_config(
            ratio=1.9,
            method="product",
            mode='ctx',
            shared_head=True,
            skip=0,
            rpe_on='qkv',
        )

        self.encoder = Attention(cfg)

        self.SSRE = SelfAttention(cfg, rpe_config=rpe_config)

        self.bef_cls_token = nn.Parameter(torch.randn(1, 1, self.att_dim))
        self.aft_cls_token = nn.Parameter(torch.randn(1, 1, self.att_dim))

        self.context1 = nn.Linear(self.att_dim, self.att_dim, bias=False)
        self.context2 = nn.Linear(self.att_dim, self.att_dim)

        self.gate1 = nn.Linear(self.att_dim, self.att_dim, bias=False)
        self.gate2 = nn.Linear(self.att_dim, self.att_dim)

        self.common_proj = nn.Linear(self.att_dim, 128)

        # self.dropout = nn.Dropout(0.5)

        self.embed = nn.Sequential(
            nn.Conv2d(self.att_dim*3, self.dim, kernel_size=1, padding=0),
            nn.GroupNorm(32, self.dim),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.att = nn.Conv2d(self.dim, 1, kernel_size=1, padding=0)
        self.fc1 = nn.Linear(self.att_dim, 6)
        # self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_1, input_2):
        batch_size, C, H, W = input_1.size()
        input_1 = input_1.view(batch_size, C, -1).permute(0, 2, 1) # (128, 196, 1026)
        input_2 = input_2.view(batch_size, C, -1).permute(0, 2, 1)
        input_bef = self.img(input_1) # (128,196, 512)
        input_aft = self.img(input_2)

        input_bef = self.SSRE(input_bef, input_bef, input_bef)
        input_aft = self.SSRE(input_aft, input_aft, input_aft)

        bef_cls_token = self.bef_cls_token.repeat(batch_size, 1, 1)
        aft_cls_token = self.aft_cls_token.repeat(batch_size, 1, 1)

        input_bef = torch.cat([bef_cls_token, input_bef], 1)
        input_aft = torch.cat([aft_cls_token, input_aft], 1)

        input_bef = self.encoder(input_bef, input_aft, input_aft)
        input_aft = self.encoder(input_aft, input_bef, input_bef)

        input_1_cls = input_bef[:, 0, :]
        input_2_cls = input_aft[:, 0, :]

        ### image-level common feature disentanglement
        bef_common = self.common_proj(input_1_cls)
        aft_common = self.common_proj(input_2_cls)
        # bef_cls_feat = F.normalize(bef_common, dim=-1)
        # aft_cls_feat = F.normalize(aft_common, dim=-1)

        bef_cls_feat = bef_common
        aft_cls_feat = aft_common

        sim_b2a = bef_cls_feat @ aft_cls_feat.t() / 0.5
        sim_a2b = aft_cls_feat @ bef_cls_feat.t() / 0.5

        sim_targets = torch.zeros_like(sim_b2a)
        sim_targets[:, :] = torch.eye(batch_size)

        loss_b2a = -torch.sum(F.log_softmax(sim_b2a, dim=1) * sim_targets, dim=1).mean()
        loss_a2b = -torch.sum(F.log_softmax(sim_a2b, dim=1) * sim_targets, dim=1).mean()
        loss_con = (loss_b2a + loss_a2b) / 2
        ##################################################

        input_bef = input_bef[:, 1:, :]
        input_aft = input_aft[:, 1:, :]

        input_diff = input_aft - input_bef

        input_bef_context = torch.tanh(self.context1(input_diff) + self.context2(input_bef))
        # input_bef_context = self.dropout(input_bef_context)
        input_bef_gate = torch.sigmoid(self.gate1(input_diff) + self.gate2(input_bef))
        # input_bef_gate = self.dropout(input_bef_gate)
        input_befs = input_bef_gate * input_bef_context

        input_aft_context = torch.tanh(self.context1(input_diff) + self.context2(input_aft))
        # input_aft_context = self.dropout(input_aft_context)
        input_aft_gate = torch.sigmoid(self.gate1(input_diff) + self.gate2(input_aft))
        # input_aft_gate = self.dropout(input_aft_gate)
        input_afts = input_aft_gate * input_aft_context

        input_bef = input_bef.permute(0, 2, 1).view(batch_size, self.att_dim, H, W)
        input_aft = input_aft.permute(0, 2, 1).view(batch_size, self.att_dim, H, W)

        input_befs = input_befs.permute(0,2,1).view(batch_size, self.att_dim, H, W)
        input_afts = input_afts.permute(0,2,1).view(batch_size, self.att_dim, H, W)
        input_diff = input_diff.permute(0,2,1).view(batch_size, self.att_dim, H, W)

        input_before = torch.cat([input_bef, input_diff, input_befs], 1)
        input_after = torch.cat([input_aft, input_diff, input_afts], 1)

        embed_before = self.embed(input_before)
        embed_after = self.embed(input_after)
        att_weight_before = torch.sigmoid(self.att(embed_before))
        att_weight_after = torch.sigmoid(self.att(embed_after))

        att_1_expand = att_weight_before.expand_as(input_bef)
        attended_1 = (input_bef * att_1_expand).sum(2).sum(2)  # (batch, dim)
        att_2_expand = att_weight_after.expand_as(input_aft)
        attended_2 = (input_aft * att_2_expand).sum(2).sum(2)  # (batch, dim)
        input_attended = attended_2 - attended_1
        pred = self.fc1(input_attended)

        return pred, att_weight_before, att_weight_after, attended_1, attended_2, input_attended, loss_con


class AddSpatialInfo(nn.Module):

    def _create_coord(self, img_feat):
        batch_size, _, h, w = img_feat.size()
        coord_map = img_feat.new_zeros(2, h, w)
        for i in range(h):
            for j in range(w):
                coord_map[0][i][j] = (j * 2.0 / w) - 1
                coord_map[1][i][j] = (i * 2.0 / h) - 1
        sequence = [coord_map] * batch_size
        coord_map_in_batch = torch.stack(sequence)
        return coord_map_in_batch

    def forward(self, img_feat):
        coord_map = self._create_coord(img_feat)
        img_feat_aug = torch.cat([img_feat, coord_map], dim=1)
        return img_feat_aug
