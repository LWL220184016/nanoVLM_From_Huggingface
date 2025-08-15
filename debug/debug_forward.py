# for debug AudioLanguageModel forward function

import os
import math
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.config import ALMConfig
from models.audio_language_model import AudioLanguageModel
from data.processors import get_audio_processor
from data.processors import get_tokenizer

def make_toy_batch(tokenizer, audio_token_id, device):
    # 構造一條包含 <AUDIO> 的簡單對話：T_user + <AUDIO> + T_assistant
    text = "User: Please transcribe the audio. <AUDIO>\nAssistant: hello world."
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
    attention_mask = torch.ones_like(input_ids, device=device)

    # labels 初始為 input_ids 的拷貝，先把「<AUDIO> 之前」全部設為 -100（只學 assistant 回覆）
    labels = input_ids.clone()
    audio_pos = (input_ids[0] == audio_token_id).nonzero(as_tuple=True)[0]
    if len(audio_pos) == 0:
        raise RuntimeError("No <AUDIO> token found in tokenized text.")
    audio_pos = int(audio_pos[0].item())
    labels[0, : audio_pos + 1] = -100  # 使用者段 + <AUDIO> 先忽略

    return input_ids, attention_mask, labels, audio_pos

def make_toy_audio(alm_cfg, device, seconds=1.0, sr=16000, freq=440.0):
    # 產生 1 秒 440Hz 正弦波
    import numpy as np
    t = np.arange(0, int(seconds * sr)) / sr
    wav = 0.1 * np.sin(2 * math.pi * freq * t).astype("float32")

    ap = get_audio_processor(alm_cfg)
    feats = ap(wav, sr).unsqueeze(0).to(device)  # [B=1, ...]
    return feats

def inspect_after_prepare(model, input_ids, audio, attention_mask, labels, tokenizer):
    # 呼叫內部展開（T_user + A + T_assistant）
    inputs_embeds, combined_attention_mask, audio_positions, audio_lens = model._prepare_decoder_inputs(
        input_ids, audio, attention_mask
    )
    B, L, D = inputs_embeds.shape
    A = int(audio_lens[0].item())
    pos = int(audio_positions[0].item())
    T_total = input_ids.shape[1]
    T_user = pos
    T_assistant = T_total - pos - 1  # 去掉 <AUDIO> 這個單一 token

    print("=== After _prepare_decoder_inputs ===")
    print(f"inputs_embeds: {inputs_embeds.shape} [B, L, D]")
    print(f"combined_attention_mask: {None if combined_attention_mask is None else combined_attention_mask.shape}")
    print(f"<AUDIO> pos: {pos}, A (audio patches): {A}")
    print(f"T_user: {T_user}, T_assistant: {T_assistant}, original T: {T_total}")
    print(f"L (expected): T_user + A + T_assistant = {T_user + A + T_assistant}")

    # 手動在 <AUDIO> 處插入 A 個 -100，得到 expanded labels（與模型 forward 一致）
    li = labels[0]
    left = li[:pos]
    right = li[pos + 1:]
    audio_ign = torch.full((A,), -100, dtype=li.dtype, device=li.device)
    li_expanded = torch.cat([left, audio_ign, right], dim=0)  # [L]
    # pad 到 batch 最長（此例單一樣本，直接比長度）
    if li_expanded.size(0) < L:
        li_expanded = F.pad(li_expanded, (0, L - li_expanded.size(0)), value=-100)
    elif li_expanded.size(0) > L:
        li_expanded = li_expanded[:L]

    print("expanded_labels length:", li_expanded.size(0))
    print("expanded_labels [-100] counts:",
          int((li_expanded[:T_user] == -100).sum().item()),
          "(T_user region) +",
          int((li_expanded[T_user:T_user + A] == -100).sum().item()),
          "(A region)")

    # 顯示對齊點附近的 token 與遮罩
    ids = input_ids[0].tolist()
    print("\n--- Raw input_ids around <AUDIO> ---")
    left_ctx = tokenizer.decode(ids[max(0, pos-15):pos], skip_special_tokens=False)
    right_ctx = tokenizer.decode(ids[pos+1:pos+16], skip_special_tokens=False)
    print(f"...{left_ctx} [<AUDIO>] {right_ctx}...")

    return inputs_embeds, combined_attention_mask, audio_positions, audio_lens, li_expanded

def run():
    

    alm_cfg = ALMConfig()
    tokenizer = get_tokenizer(alm_cfg.lm_tokenizer)
    device = alm_cfg.device
    print(f"Using device: {device}")

    # 構建模型（會自動新增 <AUDIO> 並調整詞嵌入）
    model = AudioLanguageModel(alm_cfg, load_from_HF=True, tokenizer=tokenizer, device=device).to(device)
    model.eval()

    # 造一條 toy 輸入與對應 labels（labels 僅保留 assistant 段為可學，其餘為 -100）
    input_ids, attention_mask, labels, audio_pos = make_toy_batch(tokenizer, model.audio_token_id, device)
    print("input_ids:", input_ids.shape, "labels:", labels.shape, "audio_pos:", audio_pos)

    # 造音訊特徵
    audio = make_toy_audio(alm_cfg, device)

    # 檢查 _prepare_decoder_inputs 展開後的形狀與對齊
    inputs_embeds, combined_attention_mask, audio_positions, audio_lens, expanded_labels = inspect_after_prepare(
        model, input_ids, audio, attention_mask, labels, tokenizer
    )

    # 正式跑 forward（會內部展開 labels 並計算 shift loss）
    with torch.no_grad():
        logits, loss = model(
            input_ids=input_ids,
            audio=audio,
            attention_mask=attention_mask,
            labels=labels,
        )

    print("\n=== After forward ===")
    print("logits:", logits.shape, "[B, L, V]")
    print("loss:", float(loss))

    # 驗證 logits 長度與我們手動展開 labels 長度一致
    assert logits.size(1) == expanded_labels.size(0), "Logits length != expanded labels length"

    # 驗證 expanded_labels 的前 T_user 與 A 區段都是 -100
    pos = int(audio_positions[0].item())
    A = int(audio_lens[0].item())
    T_user = pos
    assert int((expanded_labels[:T_user] == -100).all().item()) == 1, "User region should be -100"
    assert int((expanded_labels[T_user:T_user + A] == -100).all().item()) == 1, "Audio region should be -100"

    print("Checks passed: alignment and masking look correct.")

if __name__ == "__main__":
    run()