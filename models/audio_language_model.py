import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional

from models.utils import top_k_top_p_filtering
from models.audio_transformer import AudioTransformer
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
        
        # 编码音频
        audio_features = self.audio_encoder(audio)  # [B, num_patches, audio_hidden_dim]
        audio_embeds = self.MP(audio_features)  # [B, num_patches, lm_hidden_dim]
        
        # 获取文本嵌入
        text_embeds = self.decoder.token_embedding(input_ids)  # [B, seq_len, lm_hidden_dim]
        
        # 拼接音频和文本嵌入
        # 音频特征在前，文本在后
        inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1)  # [B, audio_len + seq_len, lm_hidden_dim]
        
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
            combined_attention_mask = None
        
        # 通过语言模型
        decoder_output = self.decoder(inputs_embeds, attention_mask=combined_attention_mask)
        
        # 确保通过分类头得到正确的logits
        if hasattr(self.decoder, 'head') and decoder_output.shape[-1] != self.cfg.lm_vocab_size:
            logits = self.decoder.head(decoder_output)
        else:
            logits = decoder_output
        
        # print(f"Debug - Final logits shape: {logits.shape}")
        # print(f"Debug - Expected vocab_size: {self.cfg.lm_vocab_size}")
        
        # 只对文本部分计算损失
        if targets is not None:
            # 调试信息
            # print(f"Debug - logits shape: {logits.shape}")
            # print(f"Debug - audio_embeds shape: {audio_embeds.shape}")
            # print(f"Debug - targets shape: {targets.shape}")
            # print(f"Debug - targets max: {targets.max().item()}, min: {targets.min().item()}")
            
            # 将目标序列向右移动一位用于因果语言模型
            shift_logits = logits[..., audio_embeds.shape[1]:-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            # print(f"Debug - shift_logits shape: {shift_logits.shape}")
            # print(f"Debug - shift_labels shape: {shift_labels.shape}")
            # print(f"Debug - shift_labels max: {shift_labels.max().item()}, min: {shift_labels.min().item()}")
            # print(f"Debug - vocab_size (logits dim -1): {shift_logits.size(-1)}")
            
            # # 确保shift_labels中没有超出词汇表范围的值
            # vocab_size = shift_logits.size(-1)
            # valid_mask = (shift_labels >= 0) & (shift_labels < vocab_size)
            # invalid_tokens = shift_labels[~valid_mask]
            # if len(invalid_tokens) > 0:
            #     print(f"Warning: Found invalid tokens: {invalid_tokens.unique()}")
            #     shift_labels = torch.where(valid_mask, shift_labels, -100)
            
            # 计算交叉熵损失
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
        audio_features = self.audio_encoder(audio)
        audio_embeds = self.MP(audio_features)
        
        # 获取初始文本嵌入
        text_embeds = self.decoder.token_embedding(input_ids)
        
        # 拼接音频和文本嵌入
        outputs = torch.cat([audio_embeds, text_embeds], dim=1)
        
        # 创建注意力掩码
        if attention_mask is not None:
            audio_attention_mask = torch.ones(
                batch_size, audio_embeds.shape[1], 
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            combined_attention_mask = torch.cat([audio_attention_mask, attention_mask], dim=1)
        else:
            combined_attention_mask = None
        
        # 生成新token
        generated_tokens = torch.zeros(batch_size, max_new_tokens, device=input_ids.device, dtype=torch.long)
        
        for i in range(max_new_tokens):
            # 通过语言模型
            logits = self.decoder(outputs, attention_mask=combined_attention_mask)
            
            # 获取最后一个token的logits
            last_token_logits = logits[:, -1, :]
            
            # 如果模型使用embedding模式，需要通过head层
            if not self.decoder.lm_use_tokens:
                last_token_logits = self.decoder.head(last_token_logits)
                
            if greedy:
                next_token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(last_token_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits/temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            generated_tokens[:, i] = next_token.squeeze(-1)
            
            # 将新token转换为embedding并添加到序列中
            next_embd = self.decoder.token_embedding(next_token)
            outputs = torch.cat((outputs, next_embd), dim=1)
            
            # 更新注意力掩码
            if combined_attention_mask is not None:
                new_mask = torch.ones(batch_size, 1, device=combined_attention_mask.device, dtype=combined_attention_mask.dtype)
                combined_attention_mask = torch.cat([combined_attention_mask, new_mask], dim=1)
        
        return generated_tokens

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