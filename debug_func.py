import torch

# 在训练开始前添加这个检查函数
def debug_model_dimensions(model, input_ids, audio):
    """调试模型各层的维度"""
    print("=== Model Dimension Debug ===")
    
    # 检查音频编码器
    audio_features = model.audio_encoder.encoder(audio, output_hidden_states=True)
    print(f"Audio features shape: {audio_features.shape}")
    
    # 检查模态投影器
    audio_embeds = model.MP(audio_features)
    print(f"Audio embeds shape: {audio_embeds.shape}")
    
    # 检查文本嵌入
    text_embeds = model.decoder.token_embedding(input_ids)
    print(f"Text embeds shape: {text_embeds.shape}")
    
    # 检查拼接后的嵌入
    inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1)
    print(f"Combined embeds shape: {inputs_embeds.shape}")
    
    # 检查语言模型输出
    logits = model.decoder(inputs_embeds)
    print(f"Logits shape: {logits.shape}")
    print(f"Vocab size (last dim): {logits.shape[-1]}")
    
    # 检查语言模型配置
    print(f"LM vocab size config: {model.cfg.lm_vocab_size}")
    print(f"Decoder vocab size: {getattr(model.decoder, 'vocab_size', 'Not found')}")
    
    return logits.shape[-1]

# 在训练循环开始前调用
# vocab_size = debug_model_dimensions(model, input_ids, audios)

def debug_training_step(model, input_ids, audios, attention_mask, labels):
    """调试训练步骤"""
    # 添加这些调试行：
    print(f"Batch debug - input_ids shape: {input_ids.shape}, max: {input_ids.max().item()}")
    print(f"Batch debug - labels shape: {labels.shape}, max: {labels.max().item()}")
    print(f"Batch debug - Model vocab config: {model.cfg.lm_vocab_size}")

    # 检查decoder的实际vocab_size
    if hasattr(model.decoder, 'head') and hasattr(model.decoder.head, 'out_features'):
        print(f"Decoder head in_features: {model.decoder.head.in_features}")
        print(f"Decoder head out_features: {model.decoder.head.out_features}")

# 檢查模態投影器的學習效果
def debug_modality_projection(model, audio_features, text_embeds):
    """調試模態投影器的對齊效果"""
    audio_embeds = model.MP(audio_features)
    
    # 計算音頻和文本嵌入的相似度
    audio_pooled = audio_embeds.mean(dim=1)
    text_pooled = text_embeds.mean(dim=1)
    
    # 餘弦相似度
    cos_sim = torch.cosine_similarity(audio_pooled, text_pooled, dim=-1)
    print(f"Audio-Text cosine similarity: {cos_sim.mean().item():.4f}")
    
    # 檢查嵌入分佈
    print(f"Audio embeds mean: {audio_embeds.mean().item():.4f}, std: {audio_embeds.std().item():.4f}")
    print(f"Text embeds mean: {text_embeds.mean().item():.4f}, std: {text_embeds.std().item():.4f}")

# 調試標籤掩碼
def debug_labels_masking(self, batch):
    """調試標籤掩碼是否正確"""
    for i, (input_ids, labels) in enumerate(zip(batch["input_ids"], batch["labels"])):
        print(f"Sample {i}:")
        
        # 顯示input_ids對應的文本
        input_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"Input text: {input_text}")
        
        # 顯示哪些位置被掩碼了
        masked_positions = (labels == -100).nonzero().flatten().tolist()
        print(f"Masked positions: {masked_positions}")
        
        # 顯示用於損失計算的部分
        valid_labels = labels[labels != -100]
        if len(valid_labels) > 0:
            valid_text = self.tokenizer.decode(valid_labels, skip_special_tokens=True)
            print(f"Text for loss calculation: {valid_text}")
        print("-" * 50)

