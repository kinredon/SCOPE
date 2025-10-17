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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape

from .methods_utils import submodular_function, submodular_optimizer
import numpy as np

#divprune
import os 
class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        if os.environ['BASELINE'] in ['SUBMODULE', 'CLS', 'SUBMERGE']:
            cls_attn, image_features = self.get_model().get_vision_tower()(images)
            image_features = self.get_model().mm_projector(image_features)
            return cls_attn, image_features
        else:
            image_features = self.get_model().get_vision_tower()(images)
            image_features = self.get_model().mm_projector(image_features)
            return image_features

    # divprune
    def pairwise_cosine_similarity(self, matrix):
        norm_matrix = matrix / matrix.norm(dim=1, keepdim=True)
        cosine_similarity = torch.mm(norm_matrix, norm_matrix.t())
        return cosine_similarity

    def Submodule(self, visual_feature_vectors, image_feature_length, threshold_ratio=0.1,  cls_attn=None):
        # 如果 threshold_ratio 大于 1，直接作为数量，否则按比例计算数量
        if threshold_ratio >= 1.0:
            threshold_terms = int(threshold_ratio)
        else:
            threshold_terms = int(round(threshold_ratio * image_feature_length))

        cosine_simi = self.pairwise_cosine_similarity(visual_feature_vectors)
        
        index = np.arange(image_feature_length)
        cosine_simi_np = cosine_simi.cpu().numpy()
        cls_attn = cls_attn.cpu().numpy()
        submod_function = submodular_function.__dict__[os.environ['FUNC']](index=index, cls_attn=cls_attn, similarity_matrix=cosine_simi_np)
        submod_optimizer = submodular_optimizer.__dict__['NaiveGreedy'](index=index, budget=threshold_terms, already_selected=[])

        selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
                                                        update_state=submod_function.update_state)
        
        s = torch.tensor(selection_result, dtype=torch.long, device=visual_feature_vectors.device)

        return s, cosine_simi



    def DivPrune(self, visual_feature_vectors, image_feature_length, cosine_matrix=None, threshold_ratio=0.1):  
        # import ipdb; ipdb.set_trace()          
        if threshold_ratio >= 1.0:
            threshold_terms = int(threshold_ratio)
        else:
            threshold_terms = int(round(threshold_ratio*image_feature_length))
        
        if cosine_matrix is None:
            cosine_matrix = 1.0 - (self.pairwise_cosine_similarity(visual_feature_vectors))

        s = torch.empty(threshold_terms, dtype=torch.long, device=visual_feature_vectors.device)
        for i in range(threshold_terms):
            if i==0:
                m2 = cosine_matrix
            else:
                m2 = torch.index_select(cosine_matrix, 0, torch.index_select(s,0,torch.arange(0,i,device=cosine_matrix.device)))

            if i==0:
                scores = torch.topk(m2, 2,dim=0,largest=False).values[1,:] #for distance
            else:
                scores = torch.min(m2, dim=0).values #for distance 

            phrase_to_add_idx = torch.argmax(scores)
            s[i] = phrase_to_add_idx
        return s, cosine_matrix

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0] # [576, 4096]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1) # [2, 2, 24, 24, 4096]
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous() # [4096, 2, 24, 2, 24]
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3) # [4096, 48, 48]
                            image_feature = unpad_image(image_feature, image_sizes[image_idx]) # [4096, 28, 48]
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1) # [4096, 28, 49]
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1) # [1372, 4096]
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            if os.environ['BASELINE'] in ['SUBMODULE', 'SUBMERGE', 'CLS']:
                cls_attn, image_features = self.encode_images(images)
            else:
                image_features = self.encode_images(images)
                cls_attn = None

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        #divprune
        if 'LAYER_INDEX' in os.environ:
            #print("I am called without layer 0")
            if type(image_features) == list: #this is for LLaVA 1.6
                img_feature_len = image_features[0].shape[0] #example is 2340x4096
            else: #for LLaVa 1.5
                img_feature_len = image_features.shape[1] 

            if hasattr(self.config, 'img_feature_len'):
                self.config.img_feature_len = img_feature_len
            else:
                setattr(self.config, 'img_feature_len', img_feature_len)

        if 'LAYER_INDEX' in os.environ and os.environ['LAYER_INDEX']=='0':
            SYS_TOKEN_LEN = 35 
            diverse_ratio = float(os.environ['SUBSET_RATIO']) #define the subset selection ratio
            cosine_matrix = None
            if type(image_features) == list: #this is for LLaVA 1.6
                img_feature_len = image_features[0].shape[0] #example is 2340x4096
            else: #for LLaVa 1.5
                img_feature_len = image_features.shape[1] #example is 2340x4096

            visual_tokens =new_input_embeds[0][SYS_TOKEN_LEN:SYS_TOKEN_LEN+img_feature_len]
            if os.environ['BASELINE'] == 'SUBMODULE':
                # selected_visual_tokens, cosine_matrix = self.DivPrune(visual_tokens, img_feature_len,cosine_matrix,threshold_ratio=diverse_ratio, cls_attn=cls_attn)
                selected_visual_tokens, cosine_matrix = self.Submodule(visual_tokens,img_feature_len,threshold_ratio=diverse_ratio, cls_attn=cls_attn)
            if os.environ['BASELINE'] == 'SUBMERGE':
                if int(diverse_ratio) == 64:
                    sub_number = 54
                    merge_number = 10
                elif int(diverse_ratio) == 128:
                    sub_number, merge_number = 108, 20
                elif int(diverse_ratio) == 192:
                    sub_number, merge_number = 162, 30
                elif int(diverse_ratio) == 32:
                    sub_number, merge_number = 27, 5
                
                selected_visual_tokens, cosine_matrix = self.Submodule(visual_tokens,img_feature_len,threshold_ratio=sub_number, cls_attn=cls_attn)
                remained_visual_tokens = self.filter_tokens(cosine_matrix, selected_visual_tokens)
                ones_matrix = torch.ones_like(cls_attn[remained_visual_tokens])
                r_selected, _ = self.Submodule(visual_tokens[remained_visual_tokens],len(remained_visual_tokens),threshold_ratio=merge_number, cls_attn=ones_matrix)
                r_selected_global = [remained_visual_tokens[i] for i in r_selected]
                r_selected_global = sorted(r_selected_global)
                topk = 2
                merge_visual_tokens = self.token_merge(visual_tokens, cosine_matrix, r_selected_global, remained_visual_tokens, topk)

                r_selected_global = torch.tensor(r_selected_global, device=selected_visual_tokens.device)
                selected_visual_tokens, _ = torch.cat([selected_visual_tokens, r_selected_global]).sort()
                
            elif os.environ['BASELINE'] == 'CLS':
                _, selected_visual_tokens = torch.topk(cls_attn, int(os.environ['SUBSET_RATIO']), dim=0, largest=True)  # [B, left_tokens] , sorted=True
                selected_visual_tokens = selected_visual_tokens.sort()[0]

            elif os.environ['BASELINE'] == 'OURS':
                selected_visual_tokens, cosine_matrix = self.DivPrune(visual_tokens, img_feature_len,cosine_matrix,threshold_ratio=diverse_ratio)
                      
            selected_visual_tokens += SYS_TOKEN_LEN
            keep_indexs = torch.cat((torch.arange(SYS_TOKEN_LEN,device=new_input_embeds.device), selected_visual_tokens, torch.arange(SYS_TOKEN_LEN+img_feature_len,new_input_embeds.shape[1],device=new_input_embeds.device)))
            keep_indexs = keep_indexs.sort().values

            if os.environ['BASELINE'] == 'SUBMERGE':
                # replace the token
                # import ipdb; ipdb.set_trace()
                new_input_embeds[:, r_selected_global + SYS_TOKEN_LEN] = merge_visual_tokens.to(new_input_embeds.device).unsqueeze(0)
            new_input_embeds = new_input_embeds[:,keep_indexs]
            # import ipdb; ipdb.set_trace()
            if position_ids is not None:
                position_ids = position_ids[:,keep_indexs,:]
            if attention_mask is not None:
                attention_mask = attention_mask[:,keep_indexs]
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def token_merge(self, visual_tokens, cosine_matrix, r_selected_global, remained_visual_tokens, topk=3):
        """
        Args:
            visual_tokens (torch.Tensor): (n, d), all visual token features
            cosine_matrix (torch.Tensor): (n, n), precomputed cosine similarity matrix
            r_selected_global (List[int]): selected token indices
            remained_visual_tokens (List[int]): remaining token indices (includes selected)
            topk (int): number of most similar tokens to merge with

        Returns:
            merge_visual_tokens (torch.Tensor): (len(r_selected_global), d)
        """
        device = visual_tokens.device

        n_selected = len(r_selected_global)
        n_remained = len(remained_visual_tokens)

        selected_feats = visual_tokens[r_selected_global]  # (n_selected, d)
        remained_feats = visual_tokens[remained_visual_tokens]  # (n_remained, d)

        # 提取相关的 similarity 子矩阵
        similarity = cosine_matrix[r_selected_global][:, remained_visual_tokens]  # (n_selected, n_remained)

        # 剔除自身相似度（如果 selected 出现在 remained 中）
        for idx_sel, idx_global in enumerate(r_selected_global):
            if idx_global in remained_visual_tokens:
                pos_in_remained = remained_visual_tokens.index(idx_global)
                similarity[idx_sel, pos_in_remained] = -1e3  # 极小，防止选到自己

        # 找 top-k 相似的 remained token
        topk_sim, topk_indices = similarity.topk(k=min(topk, similarity.size(1)), dim=1)  # (n_selected, topk)

        # gather topk remained features
        selected_topk_feats = remained_feats[topk_indices]  # (n_selected, topk, d)

        # 把 selected 自己也加进去
        selected_feats_expanded = selected_feats.unsqueeze(1)  # (n_selected, 1, d)
        merged_feats = torch.cat([selected_feats_expanded, selected_topk_feats], dim=1)  # (n_selected, topk+1, d)

        # 平均融合
        merge_visual_tokens = merged_feats.mean(dim=1)  # (n_selected, d)

        return merge_visual_tokens

    def filter_tokens(self, cosine_matrix, selected_visual_tokens):
        """
        Args:
            cosine_matrix (torch.Tensor): (n, n) tensor, cosine similarity matrix.
            selected_visual_tokens (List[int]): indices of selected tokens.
            
        Returns:
            remained_visual_tokens (List[int]): indices of tokens not well covered.
        """
        n = cosine_matrix.shape[0]
        device = cosine_matrix.device

        all_indices = torch.arange(n, device=device)

        # 找到未被选择的token
        unselected_mask = torch.ones(n, dtype=torch.bool, device=device)
        unselected_mask[selected_visual_tokens] = False
        unselected_indices = all_indices[unselected_mask]

        if len(selected_visual_tokens) == 0:
            # 如果没有选任何token，所有都未覆盖
            return unselected_indices.tolist()

        # Step 1: 计算 selected_visual_tokens 内部的平均相似度
        selected_matrix = cosine_matrix[selected_visual_tokens][:, selected_visual_tokens]
        # 排除对角线自己跟自己（相似度=1）
        mask = ~torch.eye(len(selected_visual_tokens), dtype=torch.bool, device=device)
        internal_similarities = selected_matrix[mask]
        internal_threshold = internal_similarities.mean()

        # Step 2: 计算未选token与已选token的平均相似度
        similarities = cosine_matrix[unselected_indices][:, selected_visual_tokens]
        avg_similarities = similarities.mean(dim=1)

        # Step 3: 过滤
        remained_indices = unselected_indices[avg_similarities < internal_threshold]

        return remained_indices.tolist()

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

    def prepare_inputs_labels_for_multimodal_pdrop(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            if os.environ['BASELINE'] in ['SUBMODULE', 'SUBMERGE', 'CLS']:
                cls_attn, image_features = self.encode_images(images)
            else:
                image_features = self.encode_images(images)
                cls_attn = None

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        image_token_posi = []
        prompt_len = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # record image position for further dropping
            image_index = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            if image_index == []:
                image_token_posi.append(-1)
            else:
                image_token_posi.append(image_index[0])
            

            # record input instruction length in inference mode
            if not self.training:  
                if image_index == []:
                    prompt_len.append(cur_input_ids.shape[0])
                else:
                    prompt_len.append(cur_input_ids.shape[0] - 1)   # consider image place holder
                                

            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0) #列表内部是原来以图片为界分成两块
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):    # 将原本text以<image>为界线分开，分别embed，并且append image feature
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        
        self.model.image_token_posi = image_token_posi
        self.model.prompt_len = prompt_len
        self.model.image_tokens = [image_feature.shape[0] for image_feature in image_features ]

        # add parameter for pdrop
        # self.model.layer_list = [2,6,16]
        # original pdrop parameters
        # self.model.image_token_ratio_list = [66,30,17] # 64
        # self.model.image_token_ratio_list = [256, 160, 88] #160
        # self.model.image_token_ratio_list = [303,110,36] # 128
        # self.model.image_token_ratio_list = [300,200,110] # 192

        # our pdrop parameters with submodule
        # self.model.image_token_ratio_list = [102, 64, 42] # 64, before 160
        # 2880 tokens
        # 160
        self.model.layer_list = [1,6,16]
        # self.model.image_token_ratio_list = [136, 88, 44] # 160
        # original: 2880*1+4*136+100*10+16*44=5128, 160*32=5120
        self.model.image_token_ratio_list = [136, 100, 44] # 160
        # 320 and 640
        # self.model.layer_list = [2,6,16]
        # self.model.image_token_ratio_list = [960, 600, 300] # 640

        # setting for FastV
        self.model.layer_list = [2]
        self.model.image_token_ratio_list = [60]  # 64 * 64 = 128 * 2 + 30 * 60

        self.model.layer_list = [3]
        self.model.image_token_ratio_list = [96]  # 192 scope, then K=3, R=0.5(96)

        self.model.image_token_ratio_list.insert(0, 1.0)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', 2048)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):  #padding
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        #divprune
        if 'LAYER_INDEX' in os.environ:
            #print("I am called without layer 0")
            if type(image_features) == list: #this is for LLaVA 1.6
                img_feature_len = image_features[0].shape[0] #example is 2340x4096
            else: #for LLaVa 1.5
                img_feature_len = image_features.shape[1] 

            if hasattr(self.config, 'img_feature_len'):
                self.config.img_feature_len = img_feature_len
            else:
                setattr(self.config, 'img_feature_len', img_feature_len)

        if 'LAYER_INDEX' in os.environ and os.environ['LAYER_INDEX']=='0':
            SYS_TOKEN_LEN = 35 
            diverse_ratio = float(os.environ['SUBSET_RATIO']) #define the subset selection ratio
            cosine_matrix = None
            if type(image_features) == list: #this is for LLaVA 1.6
                img_feature_len = image_features[0].shape[0] #example is 2340x4096
            else: #for LLaVa 1.5
                img_feature_len = image_features.shape[1] #example is 2340x4096

            visual_tokens =new_input_embeds[0][SYS_TOKEN_LEN:SYS_TOKEN_LEN+img_feature_len]
            if os.environ['BASELINE'] == 'SUBMODULE':
                # selected_visual_tokens, cosine_matrix = self.DivPrune(visual_tokens, img_feature_len,cosine_matrix,threshold_ratio=diverse_ratio, cls_attn=cls_attn)
                selected_visual_tokens, cosine_matrix = self.Submodule(visual_tokens,img_feature_len,threshold_ratio=diverse_ratio, cls_attn=cls_attn)
            if os.environ['BASELINE'] == 'SUBMERGE':
                if int(diverse_ratio) == 64:
                    sub_number = 54
                    merge_number = 10
                elif int(diverse_ratio) == 128:
                    sub_number, merge_number = 108, 20
                elif int(diverse_ratio) == 192:
                    sub_number, merge_number = 162, 30
                elif int(diverse_ratio) == 32:
                    sub_number, merge_number = 27, 5
                
                selected_visual_tokens, cosine_matrix = self.Submodule(visual_tokens,img_feature_len,threshold_ratio=sub_number, cls_attn=cls_attn)
                remained_visual_tokens = self.filter_tokens(cosine_matrix, selected_visual_tokens)
                ones_matrix = torch.ones_like(cls_attn[remained_visual_tokens])
                r_selected, _ = self.Submodule(visual_tokens[remained_visual_tokens],len(remained_visual_tokens),threshold_ratio=merge_number, cls_attn=ones_matrix)
                r_selected_global = [remained_visual_tokens[i] for i in r_selected]
                r_selected_global = sorted(r_selected_global)
                topk = 2
                merge_visual_tokens = self.token_merge(visual_tokens, cosine_matrix, r_selected_global, remained_visual_tokens, topk)

                r_selected_global = torch.tensor(r_selected_global, device=selected_visual_tokens.device)
                selected_visual_tokens, _ = torch.cat([selected_visual_tokens, r_selected_global]).sort()
                
            elif os.environ['BASELINE'] == 'CLS':
                _, selected_visual_tokens = torch.topk(cls_attn, int(os.environ['SUBSET_RATIO']), dim=0, largest=True)  # [B, left_tokens] , sorted=True
                selected_visual_tokens = selected_visual_tokens.sort()[0]

            elif os.environ['BASELINE'] == 'OURS':
                selected_visual_tokens, cosine_matrix = self.DivPrune(visual_tokens, img_feature_len,cosine_matrix,threshold_ratio=diverse_ratio)
                      
            selected_visual_tokens += SYS_TOKEN_LEN
            keep_indexs = torch.cat((torch.arange(SYS_TOKEN_LEN,device=new_input_embeds.device), selected_visual_tokens, torch.arange(SYS_TOKEN_LEN+img_feature_len,new_input_embeds.shape[1],device=new_input_embeds.device)))
            keep_indexs = keep_indexs.sort().values

            # PDROP update
            self.model.image_tokens = [diverse_ratio if image_feature.shape[0] > diverse_ratio else image_feature.shape[0] for image_feature in image_features ]

            if os.environ['BASELINE'] == 'SUBMERGE':
                # replace the token
                # import ipdb; ipdb.set_trace()
                new_input_embeds[:, r_selected_global + SYS_TOKEN_LEN] = merge_visual_tokens.to(new_input_embeds.device).unsqueeze(0)
            new_input_embeds = new_input_embeds[:,keep_indexs]
            # import ipdb; ipdb.set_trace()
            if position_ids is not None:
                position_ids = position_ids[:,keep_indexs,:]
            if attention_mask is not None:
                attention_mask = attention_mask[:,keep_indexs]

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels