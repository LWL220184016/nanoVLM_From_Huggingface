import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional

from models.utils import top_k_top_p_filtering
from models.audio_transformer import AudioTransformer_from_HF as AudioTransformer
from models.language_model import LanguageModel
from models.modality_projector import create_modality_projector
from models.config import ALMConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

class AudioLanguageModel(nn.Module):
    def __init__(self, cfg: ALMConfig, load_backbone=True, tokenizer=None, device=None):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer

        if load_backbone:
            print("Loading from backbone weights")
            self.audio_encoder = AudioTransformer(cfg, device=device)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.audio_encoder = AudioTransformer(cfg, device=device)
            self.decoder = LanguageModel(cfg)
        
        # 添加特殊 token 並調整嵌入層
        special_tokens_dict = {'additional_special_tokens': ['<AUDIO>']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids('<AUDIO>')

        self.MP = create_modality_projector(cfg)
        self.load_backbone = load_backbone
        self.device = device

    def _prepare_decoder_inputs(self, input_ids, audio, attention_mask=None):
        """
        一個內部輔助函數，封裝了從原始輸入到解碼器輸入嵌入的完整過程。
        這是 forward 和 generate 之間的共享邏輯。
        """
        batch_size = input_ids.shape[0]

        # 1. 音頻編碼和投影
        # 注意：在 generate 中，audio_encoder 可能不需要梯度
        is_training = self.training
        with torch.set_grad_enabled(is_training and self.cfg.unfreeze_audio_encoder_when_training):
            input_features = audio.to(self.device)
            audio_features = self.audio_encoder.forward(input_features, output_hidden_states=True)
        
        audio_embeds = self.MP(audio_features)  # [B, num_audio_patches, lm_hidden_dim]

        # 2. 獲取文本嵌入
        text_embeds = self.decoder.token_embedding(input_ids)  # [B, seq_len, lm_hidden_dim]

        # 3. 將音訊嵌入插入到 <AUDIO> token 的位置
        final_embeds = []
        final_attention_mask = []
        for i in range(batch_size):
            audio_token_idx = (input_ids[i] == self.audio_token_id).nonzero(as_tuple=True)[0]
            
            if len(audio_token_idx) == 0:
                # 在訓練和生成時都應報錯，因為這是數據準備階段的問題
                raise ValueError(f"<AUDIO> token not found in sample {i}. Check your data collator.")

            audio_token_idx = audio_token_idx[0]

            # 拼接嵌入
            current_embeds = torch.cat([
                text_embeds[i, :audio_token_idx],
                audio_embeds[i],
                text_embeds[i, audio_token_idx + 1:]
            ], dim=0)
            final_embeds.append(current_embeds)

            # 創建對應的注意力掩碼
            if attention_mask is not None:
                audio_attn = torch.ones(audio_embeds.shape[1], device=self.device, dtype=attention_mask.dtype)
                current_mask = torch.cat([
                    attention_mask[i, :audio_token_idx],
                    audio_attn,
                    attention_mask[i, audio_token_idx + 1:]
                ], dim=0)
                final_attention_mask.append(current_mask)

        # 4. 將 batch 中的樣本填充到相同的長度
        inputs_embeds = torch.nn.utils.rnn.pad_sequence(final_embeds, batch_first=True)
        
        combined_attention_mask = None
        if attention_mask is not None:
            combined_attention_mask = torch.nn.utils.rnn.pad_sequence(final_attention_mask, batch_first=True)
        
        return inputs_embeds, combined_attention_mask

    def forward(self, input_ids, audio, attention_mask=None, targets=None):
        """
        訓練時調用
        """
        # 調用共享函數來準備輸入
        inputs_embeds, combined_attention_mask = self._prepare_decoder_inputs(
            input_ids, audio, attention_mask
        )

        # 通过语言模型
        decoder_output_embeds = self.decoder(x=inputs_embeds, attention_mask=combined_attention_mask)
        
        try:
            logits = self.decoder.head(decoder_output_embeds[0])
        except (TypeError, IndexError):
            logits = self.decoder.head(decoder_output_embeds[1])

        if targets is not None:
            # --- 核心修改：在計算損失前，對齊 targets 的維度 ---
            # 由於我們在 _prepare_decoder_inputs 中替換了 <AUDIO> token，
            # logits 的序列長度變長了。我們需要對 targets 進行同樣的操作。
            
            batch_size = inputs_embeds.shape[0]
            aligned_targets = []
            for i in range(batch_size):
                try:
                    # 找到原始 input_ids 中的 <AUDIO> token 位置
                    audio_token_idx = (input_ids[i] == self.audio_token_id).nonzero(as_tuple=True)[0][0]
                    
                    # 獲取音訊嵌入的實際長度
                    num_audio_patches = self.cfg.audio_patches
                    
                    # 創建用於填充的 -100 標籤
                    audio_padding = torch.full((num_audio_patches,), -100, dtype=targets.dtype, device=self.device)
                    
                    # 拼接：<AUDIO>前的標籤 + 音訊填充 + <AUDIO>後的標籤
                    # 注意：我們從 targets 中移除了 <AUDIO> token 對應的那個標籤
                    current_aligned_target = torch.cat([
                        targets[i, :audio_token_idx],
                        audio_padding,
                        targets[i, audio_token_idx + 1:]
                    ], dim=0)
                    aligned_targets.append(current_aligned_target)

                except IndexError:
                    # 如果某個樣本沒有 <AUDIO> token，這是一個數據問題，但為了穩定，我們先跳過它
                    # 理想情況下，這裡應該報錯或有更複雜的處理
                    aligned_targets.append(targets[i])

            # 將對齊後的 targets 填充到與 logits 相同的長度
            # logits 的長度由 inputs_embeds 決定
            final_targets = torch.nn.utils.rnn.pad_sequence(
                aligned_targets, 
                batch_first=True, 
                padding_value=-100
            )
            
            # 確保填充後的長度與 logits 完全一致
            if final_targets.shape[1] < logits.shape[1]:
                pad_len = logits.shape[1] - final_targets.shape[1]
                final_targets = F.pad(final_targets, (0, pad_len), 'constant', -100)
            elif final_targets.shape[1] > logits.shape[1]:
                final_targets = final_targets[:, :logits.shape[1]]

            # --- 修改結束 ---

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            # 不知道爲什麽，colab 裏運行到這一行運行會報錯，換成 .view 會再報錯，再換回 .resize 就能正常運行
            loss = loss_fct(logits.resize(-1, logits.size(-1)), final_targets.resize(-1))
            return logits, loss
        
        return logits

    # 在訓練過程中添加音頻-文本對齊驗證
    def validate_audio_text_alignment(self, input_ids, audio, attention_mask=None):
        """使用檢索任務評估對齊效果"""
        self.eval()
        with torch.no_grad():
            batch_size = input_ids.shape[0]
            
            # 確保輸入在正確設備上
            input_ids = input_ids.to(self.device)
            audio = audio.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # 獲取音頻嵌入
            encoder_outputs = self.audio_encoder.forward(audio, output_hidden_states=True)
            audio_embeds = self.MP(encoder_outputs)
            
            # 獲取文本嵌入
            text_embeds = self.decoder.token_embedding(input_ids)
            
            # 音頻池化（簡單平均）
            audio_pooled = audio_embeds.mean(dim=1)  # [B, audio_dim]
            
            # 文本池化（考慮 attention_mask）
            if attention_mask is not None:
                # 使用 attention_mask 進行加權平均
                mask_expanded = attention_mask.unsqueeze(-1).expand(text_embeds.size()).float()
                sum_embeddings = torch.sum(text_embeds * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                text_pooled = sum_embeddings / sum_mask  # [B, text_dim]
            else:
                text_pooled = text_embeds.mean(dim=1)  # [B, text_dim]
            
            # 檢查維度是否匹配
            if audio_pooled.shape[-1] != text_pooled.shape[-1]:
                # 如果維度不匹配，需要投影到相同維度
                if not hasattr(self, 'alignment_projection'):
                    # 創建投影層（這應該在模型初始化時做）
                    min_dim = min(audio_pooled.shape[-1], text_pooled.shape[-1])
                    self.audio_proj = nn.Linear(audio_pooled.shape[-1], min_dim).to(self.device)
                    self.text_proj = nn.Linear(text_pooled.shape[-1], min_dim).to(self.device)
                
                audio_pooled = self.audio_proj(audio_pooled)
                text_pooled = self.text_proj(text_pooled)
            
            # 歸一化特徵
            audio_pooled = F.normalize(audio_pooled, p=2, dim=-1)
            text_pooled = F.normalize(text_pooled, p=2, dim=-1)
            
            # 計算所有可能的音頻-文本對的相似度
            similarities = torch.matmul(audio_pooled, text_pooled.T)  # [B, B]
            
            # 計算檢索準確率
            # 對於每個音頻，看是否能正確檢索到對應文本
            audio_to_text_acc = (similarities.argmax(dim=1) == torch.arange(batch_size).to(self.device)).float().mean()
            
            # 對於每個文本，看是否能正確檢索到對應音頻
            text_to_audio_acc = (similarities.argmax(dim=0) == torch.arange(batch_size).to(self.device)).float().mean()
            
            # 返回雙向檢索的平均準確率
            return (audio_to_text_acc + text_to_audio_acc).item() / 2

    # 在生成時添加調試信息
    @torch.no_grad()
    def generate_with_debug(self, input_ids, audio, attention_mask=None, max_new_tokens=5, **kwargs):
        """帶調試信息的生成方法"""
        # 檢查音頻-文本對齊
        alignment_score = self.validate_audio_text_alignment(input_ids, audio, attention_mask)
        print(f"Audio-Text alignment score: {alignment_score:.4f}")
        
        # 原始生成邏輯
        return self.generate(input_ids, audio, attention_mask, max_new_tokens, **kwargs)

    @torch.no_grad()
    def generate(self, input_ids, audio, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False):
        """音频问答生成"""
        self.eval()
        batch_size = input_ids.shape[0]
        
        # 調用共享函數來準備初始輸入
        current_sequence_embeds, current_attention_mask = self._prepare_decoder_inputs(
            input_ids, audio, attention_mask
        )
        
        generated_tokens_ids_list = [] 
        
        for _ in range(max_new_tokens):
            # 注意：這裡傳入的是已經準備好的 embeds
            outputs = self.decoder.decoder(inputs_embeds=current_sequence_embeds, attention_mask=current_attention_mask)
            decoder_output_embeds = outputs.last_hidden_state
            
            last_token_embed_from_decoder = decoder_output_embeds[:, -1, :]
            last_token_logits = self.decoder.head(last_token_embed_from_decoder)
                
            if greedy:
                next_token_id = torch.argmax(last_token_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(last_token_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            
            generated_tokens_ids_list.append(next_token_id)
            
            # 檢查是否生成了 EOS token
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            next_token_embed = self.decoder.token_embedding(next_token_id)
            current_sequence_embeds = torch.cat((current_sequence_embeds, next_token_embed), dim=1)
            
            new_mask_segment = torch.ones(batch_size, 1, device=current_attention_mask.device, dtype=current_attention_mask.dtype)
            current_attention_mask = torch.cat([current_attention_mask, new_mask_segment], dim=1)
        
        generated_tokens_tensor = torch.cat(generated_tokens_ids_list, dim=1)
        return generated_tokens_tensor

    # 其他方法（save_pretrained, from_pretrained, push_to_hub）可以参考原来的AudioLanguageModel实现
    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, tokenizer
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
            print("=================================================")
            print("load model from local")
            print("=================================================")
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
            raise ValueError(
                f"Path {repo_id_or_path} not exist. Please provide a valid path."
            )

        # Load config
        with open(config_path, "r") as f:
            cfg = ALMConfig(**json.load(f))

        # Initialize model without loading the backbone
        model = cls(cfg, load_backbone=False, tokenizer=tokenizer)

        # Load safetensors weights
        load_model(model, weights_path, strict=False)
        
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