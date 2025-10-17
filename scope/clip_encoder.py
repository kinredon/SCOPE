#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2024 Senqiao Yang
# ------------------------------------------------------------------------
# Modified from VisionZip (https://github.com/dvlab-research/VisionZip)
# Copyright 2024 Jinhong Deng
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import glob

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention, CLIPEncoder

from .utils import CLIPAttention_forward, CLIP_EncoderLayer_forward

sys.path.append('/mnt/data1/yalun/djh_workspace/divprune/LLaVA/llava/model/methods_utils')
import submodular_function, submodular_optimizer


class CLIPVisionTower_VisionZip(nn.Module):


    @torch.no_grad()
    def forward(self, images):
        import time
        # start_time = time.time()
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, output_attentions=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
            attn_weights  = image_forward_outs.attentions[-2]
            hidden_states = image_forward_outs.hidden_states[-2]
            metric = self.vision_tower.vision_model.encoder.layers[-2].metric
            dominant_num =  int(self.vision_tower._info["dominant"] / images.shape[0] + 0.5)
            contextual_num = int(self.vision_tower._info["contextual"] / images.shape[0] + 0.5)

            ## Dominant Visual Tokens
            cls_idx = 0
            cls_attention = attn_weights[:, :, cls_idx, cls_idx+1:]  
            cls_attention_sum = cls_attention.sum(dim=1)  
            topk_indices = cls_attention_sum.topk(dominant_num, dim=1).indices + 1
            all_indices = torch.cat([torch.zeros((hidden_states.shape[0], 1), dtype=topk_indices.dtype, device=topk_indices.device), topk_indices], dim=1)
            
            mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(1, all_indices, False)
            dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], dominant_num + 1, hidden_states.shape[2])
            
            ### Filter
            metric_filtered = metric[mask].view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num + 1), metric.shape[2])

            hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num +1), hidden_states.shape[2])  
            
            metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True) 

            ## Contextual Visual Tokens
            step = max(1, metric_normalized.shape[1] // contextual_num)
            target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_num]
            target_tokens = metric_normalized[:, target_indices, :]

            tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1], device=metric_normalized.device), target_indices), :]
            similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
            assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], contextual_num, dtype=hidden_states_filtered.dtype, device=metric_normalized.device)
            assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
            counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
            hidden_to_merge = hidden_states_filtered[:, ~torch.isin(torch.arange(hidden_states_filtered.shape[1], device=hidden_states_filtered.device), target_indices), :]
            aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
            target_hidden = hidden_states_filtered[:, target_indices, :]  
            
            contextual_tokens = target_hidden + aggregated_hidden

            # Merge with target hidden states and concatenate
            hidden_states_save = torch.cat([dominant_tokens, contextual_tokens], dim=1).to(images.dtype)

        # end_time = time.time()
        # print(f"Time taken for VisionZip: {end_time - start_time} seconds")
        return hidden_states_save, all_indices


class CLIPVisionTower_SCOPE(nn.Module):

    @torch.no_grad()
    def forward(self, images):
        # import time
        # start_time = time.time()
        # import ipdb; ipdb.set_trace()
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, output_attentions=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
            attn_weights  = image_forward_outs.attentions[-2]
            hidden_states = image_forward_outs.hidden_states[-2]
            metric = self.vision_tower.vision_model.encoder.layers[-2].metric
            dominant_num =  self.vision_tower._info["dominant"]

            cls_idx = 0
            cls_attention = attn_weights[:, :, cls_idx, cls_idx+1:]
            cls_attention_sum = cls_attention.sum(dim=1)

            image_features = hidden_states[:, cls_idx + 1:]
            bs = image_features.shape[0]
            dominant_num = int(dominant_num /bs)
            selected_idx, _ = SCOPE(image_features, dominant_num, cls_attention_sum)
            selected_idx += 1

            all_indices = selected_idx 
            mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(1, all_indices, False)
            dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], dominant_num, hidden_states.shape[2])
            
            hidden_states_save = dominant_tokens

        return hidden_states_save, all_indices

def SCOPE(visual_feature_vectors, num_selected_token, cls_attn=None):
    """
    Batched version of SCOPE that processes all batch elements simultaneously.
    Args:
        visual_feature_vectors: [B, N, D] batch of feature vectors
        num_selected_token: Number of tokens to select per batch
        cls_attn: [B, N] batch of attention weights
    Returns:
        selected_idx: [B, K] selected token indices for each batch
        cosine_simi: [B, N, N] batch of cosine similarity matrices
    """
    # Calculate cosine similarity for all batches at once
    norm_vectors = visual_feature_vectors / visual_feature_vectors.norm(dim=-1, keepdim=True)
    cosine_simi = torch.bmm(norm_vectors, norm_vectors.transpose(1, 2))
    
    B, N = visual_feature_vectors.shape[:2]
    device = visual_feature_vectors.device
    dtype = visual_feature_vectors.dtype
    
    # Pre-allocate tensors for all batches
    selected = torch.zeros(B, N, dtype=torch.bool, device=device)
    selected_idx = torch.empty(B, num_selected_token, dtype=torch.long, device=device)
    cur_max = torch.zeros(B, N, dtype=dtype, device=device)
    
    # Precompute cls_attn ** alpha for all batches
    alpha = float(os.environ.get('ALPHA', '1.0'))
    if cls_attn is not None:
        cls_attn_powered = cls_attn ** alpha
    else:
        cls_attn_powered = torch.ones(B, N, dtype=dtype, device=device)
    
    for i in range(num_selected_token):
        # Calculate gains for all batches simultaneously
        unselected_mask = ~selected
        gains = torch.maximum(
            torch.zeros(1, dtype=dtype, device=device),
            cosine_simi.masked_fill(~unselected_mask.unsqueeze(1), 0) - 
            cur_max.unsqueeze(2)
        ).sum(dim=1)
        
        # Apply attention weights
        combined = os.environ.get('COMBINED', 'multi')
        if combined == 'multi':
            gains = gains * cls_attn_powered
        elif combined == 'add':
            gains = gains + cls_attn_powered
        else:
            raise NotImplementedError
        # Mask out already selected tokens
        gains = gains.masked_fill(~unselected_mask, float('-inf'))
        
        # Find best elements for all batches
        best_idx = gains.argmax(dim=1)
        
        # Update states for all batches
        selected[torch.arange(B, device=device), best_idx] = True
        selected_idx[:, i] = best_idx
        cur_max = torch.maximum(cur_max, cosine_simi[torch.arange(B, device=device), best_idx])
    
    return selected_idx, cosine_simi



