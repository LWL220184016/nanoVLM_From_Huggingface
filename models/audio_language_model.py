import json
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import asdict
from typing import Optional

from models.utils import top_k_top_p_filtering
from models.audio_transformer import AudioTransformer_from_HF as AudioTransformer
from models.language_model import LanguageModel
from models.modality_projector import create_modality_projector
from models.config import ALMConfig

from safetensors.torch import load_model, save_model
from debug.debug_func import debug_print_tensor_stats

class AudioLanguageModel(nn.Module):
    def __init__(self, cfg: ALMConfig, load_from_HF=True, tokenizer=None, print_debug=False):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.print_debug = print_debug
        self.device = cfg.device

        self.audio_encoder = AudioTransformer(cfg, load_from_HF=load_from_HF)
        if load_from_HF:
            print("Loading from backbone weights")
            self.decoder = LanguageModel.from_huggingface_pretrained(cfg).to(self.device)
        else:
            self.decoder = LanguageModel(cfg).to(self.device)
                
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        self.MP = create_modality_projector(cfg).to(self.device)
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids('<AUDIO>')

    def _prepare_decoder_inputs(self, input_ids, audio, attention_mask=None):
        """
        準備解碼器輸入。回傳：
        - inputs_embeds, combined_attention_mask
        - audio_positions: 每筆樣本 <AUDIO> 的索引（在原始 input_ids 中）
        - audio_lens: 每筆樣本展開後音訊片段數 A
        """
        input_ids = input_ids.to(self.device)
        audio = audio.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        batch_size = input_ids.shape[0]

        is_training = self.training
        with torch.set_grad_enabled(is_training and self.cfg.unfreeze_audio_encoder_when_training):
            audio_features = self.audio_encoder.forward(audio, output_hidden_states=True)

        audio_embeds = self.MP(audio_features)  # [B, A, D]
        text_embeds = self.decoder.token_embedding(input_ids)  # [B, T, D]
        
        # Debug
        if self.print_debug:
            print("Debug(AudioLanguageModel): text_embeds: ", text_embeds.size())  # 調試輸出
            print("Debug(AudioLanguageModel): audio_embeds: ", audio_embeds.size())  # 調試輸出


        # 尋找所有 <AUDIO> token 的位置
        audio_token_mask = (input_ids == self.audio_token_id)
        audio_positions = torch.where(audio_token_mask)[1]
        
        # 檢查每個樣本是否都有 <AUDIO> token
        if not torch.all(audio_token_mask.sum(dim=1) == 1):
            raise ValueError("Each sample in the batch must contain exactly one <AUDIO> token.")

        A = audio_embeds.shape[1] # 音訊片段數
        D = audio_embeds.shape[2] # embedding 維度
        T = text_embeds.shape[1]  # 原始文本長度
        L = T - 1 + A             # 拼接後的總長度

        # 創建一個新的 embedding tensor 和 attention mask
        final_embeds = torch.zeros(batch_size, L, D, device=self.device, dtype=text_embeds.dtype)
        final_attention_mask = torch.zeros(batch_size, L, device=self.device, dtype=attention_mask.dtype) if attention_mask is not None else None

        # 創建索引來高效地填充張量
        arange_batch = torch.arange(batch_size, device=self.device)
        arange_L = torch.arange(L, device=self.device)
        
        # 填充 <AUDIO> 左側的文本
        mask_before = arange_L.unsqueeze(0) < audio_positions.unsqueeze(1)
        indices_before = torch.where(mask_before)
        final_embeds[indices_before] = text_embeds[indices_before[0], indices_before[1]]
        if final_attention_mask is not None:
            final_attention_mask[indices_before] = attention_mask[indices_before[0], indices_before[1]]

        # 填充音訊部分
        mask_audio = (arange_L.unsqueeze(0) >= audio_positions.unsqueeze(1)) & (arange_L.unsqueeze(0) < (audio_positions + A).unsqueeze(1))
        indices_audio = torch.where(mask_audio)
        final_embeds[indices_audio] = audio_embeds.view(-1, D)[(indices_audio[0] * A + (indices_audio[1] - audio_positions[indices_audio[0]]))]
        if final_attention_mask is not None:
            final_attention_mask[indices_audio] = 1

        # 填充 <AUDIO> 右側的文本
        mask_after = arange_L.unsqueeze(0) >= (audio_positions + A).unsqueeze(1)
        indices_after = torch.where(mask_after)
        original_indices_after = indices_after[1] - A + 1
        final_embeds[indices_after] = text_embeds[indices_after[0], original_indices_after]
        if final_attention_mask is not None:
            final_attention_mask[indices_after] = attention_mask[indices_after[0], original_indices_after]

        audio_lens = torch.full((batch_size,), A, device=self.device, dtype=torch.long)
        return final_embeds, final_attention_mask, audio_positions, audio_lens

    def forward(self, input_ids, audio, attention_mask=None, labels=None):
        """
        訓練時調用：
        - 若 labels 是「答案Only」(音頻轉錄的token序列)，我們不在中間插入 -100，而是將整段答案
          連續對齊到展開後序列的尾端（或你指定的起點），其餘位置以 -100 忽略
        - 使用 next-token shift 計算損失
        """
        inputs_embeds, combined_attention_mask, audio_positions, audio_lens = self._prepare_decoder_inputs(
            input_ids, audio, attention_mask
        )

        decoder_output_embeds, _ = self.decoder(x=inputs_embeds, attention_mask=combined_attention_mask)
        logits = self.decoder.head(decoder_output_embeds)  # [B, L, V]

        if labels is not None:
            B = input_ids.size(0)
            L = logits.size(1)
            expanded_labels = []

            for i in range(B):
                pos = int(audio_positions[i].item())   # <AUDIO> 在原始 input_ids 的位置
                A   = int(audio_lens[i].item())        # 插入的音頻片段長度
                li  = labels[i]                        # 若是「答案Only」，shape 為 [S]

                # 構造展開後長度的 base 標籤，先全部忽略
                li_expanded = torch.full((L,), -100, dtype=li.dtype, device=li.device)

                # 將整段答案連續貼在序列尾端（避免把答案切斷）
                # 你也可以改成：start = pos + A + right_prompt_len（若你知道音頻後的文字前綴長度）
                S = li.numel()
                start = max(0, L - S)
                li_expanded[start:start + S] = li

                expanded_labels.append(li_expanded)

                # Debug
                if self.print_debug:
                    debug_print_tensor_stats("Debug(AudioLanguageModel): Input Embeds = \n", inputs_embeds) # <--- 調試點 1
                    debug_print_tensor_stats("Debug(AudioLanguageModel): Decoder Output Embeds = \n", decoder_output_embeds) # <--- 調試點 1
                    
                    print("Debug(AudioLanguageModel): input_ids: ", input_ids.size(), "input_ids: ", input_ids)  # 調試輸出
                    print("Debug(AudioLanguageModel): labels: ", li.size(), "li: ", li)  # 調試輸出
                    print(f"Debug(AudioLanguageModel): <AUDIO> pos: {pos}, A (audio patches): {audio_lens[i].item()}")
                    print("Debug(AudioLanguageModel): A: ", A)
                    print("Debug(AudioLanguageModel):   S: ",  S.size(), "   S: ",   S)
                    print("Debug(AudioLanguageModel): start: ", start.size(), "start: ", start)
                    # print("Debug(AudioLanguageModel): audio_ign: ", audio_ign.size(), "audio_ign: ", audio_ign)
                    print("Debug(AudioLanguageModel): li_expanded: ", li_expanded.size(), "li_expanded: ", li_expanded)

            combined_labels = torch.stack(expanded_labels, dim=0)  # [B, L]

            # next-token shift
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = combined_labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
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
        """音频问答生成 (使用 KV Cache 提升效率)"""
        self.eval()
        
        # 準備初始輸入
        inputs_embeds, combined_attention_mask, _, _ = self._prepare_decoder_inputs(
            input_ids, audio, attention_mask
        )
        
        # --- 建議修改 4: 使用 KV Cache ---
        # 第一次 forward pass，獲取 past_key_values
        outputs = self.decoder(x=inputs_embeds, attention_mask=combined_attention_mask, use_cache=True)
        past_key_values = outputs.past_key_values
        
        # 從最後一個 token 的 logits 開始生成
        last_token_logits = self.decoder.head(outputs.last_hidden_state[:, -1, :])
        
        generated_tokens_ids_list = []

        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(last_token_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(last_token_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            
            generated_tokens_ids_list.append(next_token_id)
            
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            # 後續的 forward pass 只需傳入新的 token_id 和 past_key_values
            outputs = self.decoder(
                input_ids=next_token_id, 
                attention_mask=combined_attention_mask, # attention_mask 需要更新
                past_key_values=past_key_values, 
                use_cache=True
            )
            
            last_token_logits = self.decoder.head(outputs.last_hidden_state[:, -1, :])
            past_key_values = outputs.past_key_values
            
            # 更新 attention_mask
            new_mask_segment = torch.ones((combined_attention_mask.shape[0], 1), device=self.device, dtype=combined_attention_mask.dtype)
            combined_attention_mask = torch.cat([combined_attention_mask, new_mask_segment], dim=1)

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
        model = cls(cfg, load_from_HF=False, tokenizer=tokenizer)

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