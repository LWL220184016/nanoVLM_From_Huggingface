import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional

from models.utils import top_k_top_p_filtering
from models.audio_transformer import AudioTransformer_from_HF as AudioTransformer
from models.language_model import LanguageModel
from models.modality_projector import ModalityProjector
from models.config import ALMConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

class AudioLanguageModel(nn.Module):
    def __init__(self, cfg: ALMConfig, load_backbone=True):
        super().__init__()
        self.cfg = cfg
        if load_backbone:
            print("Loading from backbone weights")
            self.audio_encoder = AudioTransformer.from_pretrained(cfg)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.audio_encoder = AudioTransformer(cfg)
            self.decoder = LanguageModel(cfg)
        self.MP = ModalityProjector(cfg)
        self.load_backbone = load_backbone

    def forward(self, input_ids, audio, attention_mask=None, targets=None):
        """
        input_ids: [batch_size, seq_length] 文本token
        audio: [batch_size, audio_length] 原始音频波形
        attention_mask: [batch_size, seq_length] 注意力掩码
        targets: [batch_size, seq_length] 目标token（用于训练）
        """
        batch_size = input_ids.shape[0]
    
        # 编码音频 - 添加記憶體優化
        with torch.no_grad():  # 在音頻編碼時關閉梯度計算
            audio_features = self.audio_encoder.encoder(audio, output_hidden_states=True)
            audio_embeddings = audio_features.last_hidden_state.detach()  # 分離梯度
        
        # 重新啟用梯度用於模態投影器
        audio_embeddings.requires_grad_(True)
        audio_embeds = self.MP(audio_embeddings)  # [B, num_patches, lm_hidden_dim]
        
        # 获取文本嵌入
        text_embeds = self.decoder.token_embedding(input_ids)  # [B, seq_len, lm_hidden_dim]
        
        # 拼接音频和文本嵌入
        inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1)
        
        # 创建组合的注意力掩码
        if attention_mask is not None:
            # 为音频部分创建全1的掩码
            audio_attention_mask = torch.ones(
                batch_size, audio_embeds.shape[1], 
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            # 拼接音频和文本的注意力掩码
            combined_attention_mask = torch.cat([audio_attention_mask, attention_mask], dim=1)
        else:
            # 如果原始 attention_mask 為 None，則為整個序列創建一個全1的掩碼
            combined_attention_mask = torch.ones(
                batch_size, inputs_embeds.shape[1],
                device=inputs_embeds.device, dtype=torch.long
            )
        
        # 通过语言模型 (self.cfg.lm_use_tokens is False, so self.decoder() returns embeddings)
        decoder_output_embeds = self.decoder(inputs_embeds, attention_mask=combined_attention_mask)
        
        # 必须通过 head 获取 logits
        logits = self.decoder.head(decoder_output_embeds)
        
        # 只对文本部分计算损失
        if targets is not None:
            # logits 的文本部分从 audio_embeds.shape[1] 开始
            text_logits = logits[:, audio_embeds.shape[1]:, :] 
            # Causal LM loss: 预测下一个 token, 所以 logits 和 labels 需要移位
            shift_logits = text_logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous() # targets 对应原始的 input_ids (文本部分)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            return logits, loss
        
        return logits

    @torch.no_grad()
    def generate(self, input_ids, audio, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False):
        """音频问答生成"""
        self.eval()
        batch_size = input_ids.shape[0]
        
        # 编码音频
        audio_features = self.audio_encoder.encoder(audio, output_hidden_states=True) # [B, num_patches, audio_hidden_dim]
        audio_embeddings = audio_features.last_hidden_state
        audio_embeds = self.MP(audio_embeddings)  # [B, num_patches, lm_hidden_dim]
        
        # 获取初始文本嵌入
        text_embeds = self.decoder.token_embedding(input_ids)
        
        # 拼接音频和文本嵌入
        current_sequence_embeds = torch.cat([audio_embeds, text_embeds], dim=1)
        
        # 创建注意力掩码
        if attention_mask is not None: 
            audio_attention_mask = torch.ones(
                batch_size, audio_embeds.shape[1], 
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            current_attention_mask = torch.cat([audio_attention_mask, attention_mask], dim=1)
        else: 
            current_attention_mask = torch.ones(
                batch_size, current_sequence_embeds.shape[1], 
                device=current_sequence_embeds.device, dtype=torch.long
            )
        
        generated_tokens_ids_list = [] 
        
        for _ in range(max_new_tokens):
            # 通过语言模型 (self.decoder.lm_use_tokens is False, so self.decoder() returns embeddings)
            decoder_output_embeds = self.decoder(current_sequence_embeds, attention_mask=current_attention_mask)
            
            # 获取最后一个token的输出嵌入 (对应下一个token的预测)
            last_token_embed_from_decoder = decoder_output_embeds[:, -1, :]
            
            # 通过 head 层得到 logits
            last_token_logits = self.decoder.head(last_token_embed_from_decoder)
                
            if greedy:
                next_token_id = torch.argmax(last_token_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(last_token_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            
            generated_tokens_ids_list.append(next_token_id)
            
            # 将新token转换为embedding并添加到序列中，为下一次迭代做准备
            next_token_embed = self.decoder.token_embedding(next_token_id)
            current_sequence_embeds = torch.cat((current_sequence_embeds, next_token_embed), dim=1)
            
            # 更新注意力掩码
            new_mask_segment = torch.ones(batch_size, 1, device=current_attention_mask.device, dtype=current_attention_mask.dtype)
            current_attention_mask = torch.cat([current_attention_mask, new_mask_segment], dim=1)
        
        generated_tokens_tensor = torch.cat(generated_tokens_ids_list, dim=1)
        return generated_tokens_tensor

    # 其他方法（save_pretrained, from_pretrained, push_to_hub）可以参考原来的AudioLanguageModel实现
    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: Optional[str] = None
    ) -> "AudioLanguageModel":
        """
        Load a VisionLanguageModel from a local directory or a repo on the Hugging Face Hub.

        Args:
            repo_id_or_path (str): The path to the local directory or the Hugging Face Hub repo ID.

        Returns:
            VisionLanguageModel: The loaded model.
        """
        # If local folder exists => load from there
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            if not os.path.exists(weights_path):
                raise ValueError(
                    f"Weights file not found at {weights_path}. Please provide a valid path."
                )
        # Otherwise, assume it's a Hugging Face Hub repo
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        # Load config
        with open(config_path, "r") as f:
            cfg = ALMConfig(**json.load(f))

        # Initialize model without loading the backbone
        model = cls(cfg, load_backbone=False)

        # Load safetensors weights
        load_model(model, weights_path)

        # Done!
        return model

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and configuration to a directory.

        Args:
            save_directory (str): The directory to save the model and config.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        # Save weights as safetensors
        save_model(self, os.path.join(save_directory, "model.safetensors"))