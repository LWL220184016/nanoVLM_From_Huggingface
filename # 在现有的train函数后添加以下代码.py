# 在现有的train函数后添加以下代码

def contrastive_loss(audio_embeds, text_embeds, temperature=0.07):
    """对比学习损失"""
    # 池化到固定维度
    audio_pooled = audio_embeds.mean(dim=1)  # [B, hidden_dim]
    text_pooled = text_embeds.mean(dim=1)    # [B, hidden_dim]
    
    # 归一化
    audio_pooled = F.normalize(audio_pooled, dim=-1)
    text_pooled = F.normalize(text_pooled, dim=-1)
    
    # 计算相似度矩阵
    logits = torch.matmul(audio_pooled, text_pooled.T) / temperature
    
    # 对角线为正样本
    labels = torch.arange(logits.size(0)).to(logits.device)
    
    # 双向对比损失
    loss_a2t = F.cross_entropy(logits, labels)
    loss_t2a = F.cross_entropy(logits.T, labels)
    
    return (loss_a2t + loss_t2a) / 2

def train_step1_alignment(train_cfg, alm_cfg):
    """第一步：模态投影器对齐训练"""
    print("=== Stage 1: Modality Projector Alignment ===")
    
    train_loader, val_loader, test_loader = get_dataloaders(train_cfg, alm_cfg)
    tokenizer = get_tokenizer(alm_cfg.lm_tokenizer)

    # Initialize model
    model = AudioLanguageModel(alm_cfg)
    print(f"nanoALM initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # 冻结音频编码器和语言模型
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False
    
    print(f"Stage 1: Only training {sum(p.numel() for p in model.MP.parameters() if p.requires_grad):,} MP parameters")

    # 只优化模态投影器
    optimizer = optim.AdamW(model.MP.parameters(), lr=train_cfg.lr_mp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if train_cfg.compile:
        model = torch.compile(model)

    batch_losses = []
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(train_cfg.stage1_epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1}"):
            audios = batch["audio"].to(device)
            input_ids = batch["input_ids"].to(device)
            
            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # 提取特征
                audio_features = model.audio_encoder(audios)
                audio_embeds = model.MP(audio_features)
                
                # 获取文本embedding (不包含最后一个token)
                text_embeds = model.decoder.token_embedding(input_ids[:, :-1])
                
                # 对比学习损失
                loss = contrastive_loss(audio_embeds, text_embeds)

            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_train_loss += batch_loss
            batch_losses.append(batch_loss)

            if global_step % 50 == 0:
                print(f"Stage1 Step: {global_step}, Alignment Loss: {batch_loss:.4f}")

            global_step += 1

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Stage1 Epoch {epoch+1}/{train_cfg.stage1_epochs} | Alignment Loss: {avg_train_loss:.4f}")
        
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            model.save_pretrained(save_directory=f"{alm_cfg.alm_checkpoint_path}/stage1_best")

    # 保存第一阶段模型
    model.save_pretrained(save_directory=f"{alm_cfg.alm_checkpoint_path}/stage1_final")
    print("Stage 1 completed!")
    plt.plot(batch_losses, label='Train Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return model

def train_step2_pretraining(train_cfg, alm_cfg, stage1_model=None):
    """第二步：语言模型预训练"""
    print("=== Stage 2: Language Model Pretraining ===")
    
    train_loader, val_loader, test_loader = get_dataloaders(train_cfg, alm_cfg)
    tokenizer = get_tokenizer(alm_cfg.lm_tokenizer)

    # 加载第一阶段模型或从头开始
    if stage1_model is not None:
        model = stage1_model
    else:
        try:
            model = AudioLanguageModel.from_pretrained(f"{alm_cfg.alm_checkpoint_path}/stage1_final")
            print("Loaded Stage 1 model")
        except:
            model = AudioLanguageModel(alm_cfg)
            print("Starting Stage 2 from scratch")

    # 冻结音频编码器，解冻语言模型和模态投影器
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = True
    for param in model.MP.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 2: Training {trainable_params:,} parameters")

    # 不同学习率
    param_groups = [
        {'params': model.MP.parameters(), 'lr': train_cfg.lr_mp * 0.1},
        {'params': model.decoder.parameters(), 'lr': train_cfg.lr_backbones}
    ]
    optimizer = optim.AdamW(param_groups)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if train_cfg.compile:
        model = torch.compile(model)

    batch_losses = []
    val_losses = []
    val_plot_steps = []
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(train_cfg.stage2_epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Stage2 Epoch {epoch+1}"):
            audios = batch["audio"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # 使用因果语言建模损失
                _, loss = model(input_ids, audios, attention_mask=attention_mask, targets=labels)

            loss.backward()

            # 动态学习率调整
            adj_lr_mp = get_lr(global_step, train_cfg.lr_mp * 0.1, len(train_loader) * train_cfg.stage2_epochs)
            adj_lr_backbones = get_lr(global_step, train_cfg.lr_backbones, len(train_loader) * train_cfg.stage2_epochs)
            optimizer.param_groups[0]['lr'] = adj_lr_mp
            optimizer.param_groups[1]['lr'] = adj_lr_backbones

            optimizer.step()

            batch_loss = loss.item()
            total_train_loss += batch_loss
            batch_losses.append(batch_loss)

            if global_step % 50 == 0:
                print(f"Stage2 Step: {global_step}, Pretraining Loss: {batch_loss:.4f}")

            global_step += 1

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Stage2 Epoch {epoch+1}/{train_cfg.stage2_epochs} | Pretraining Loss: {avg_train_loss:.4f}")
        
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            model.save_pretrained(save_directory=f"{alm_cfg.alm_checkpoint_path}/stage2_best")

    # 保存第二阶段模型
    model.save_pretrained(save_directory=f"{alm_cfg.alm_checkpoint_path}/stage2_final")
    print("Stage 2 completed!")
    plt.plot(batch_losses, label='Train Loss')
    plt.plot(val_plot_steps, val_losses, label='Val Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.show()

    return model

def train_step3_instruction_tuning(train_cfg, alm_cfg, stage2_model=None):
    """第三步：指令微调"""
    print("=== Stage 3: Instruction Tuning ===")
    
    train_loader, val_loader, test_loader = get_dataloaders(train_cfg, alm_cfg)
    tokenizer = get_tokenizer(alm_cfg.lm_tokenizer)

    # 加载第二阶段模型
    if stage2_model is not None:
        model = stage2_model
    else:
        try:
            model = AudioLanguageModel.from_pretrained(f"{alm_cfg.alm_checkpoint_path}/stage2_final")
            print("Loaded Stage 2 model")
        except:
            print("No Stage 2 model found, using current model")
            model = AudioLanguageModel(alm_cfg)

    # 全部解冻，使用较小学习率
    for param in model.parameters():
        param.requires_grad = True

    print(f"Stage 3: Training all {sum(p.numel() for p in model.parameters()):,} parameters")

    # 更小的学习率
    param_groups = [
        {'params': model.MP.parameters(), 'lr': train_cfg.lr_mp * 0.01},
        {'params': model.decoder.parameters(), 'lr': train_cfg.lr_backbones * 0.1},
        {'params': model.audio_encoder.parameters(), 'lr': train_cfg.lr_backbones * 0.01}
    ]
    optimizer = optim.AdamW(param_groups)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if train_cfg.compile:
        model = torch.compile(model)

    # 这里可以使用原来的训练循环，但数据应该是指令格式
    # 暂时使用相同的数据格式
    best_accuracy = 0
    global_step = 0
    
    for epoch in range(train_cfg.stage3_epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Stage3 Epoch {epoch+1}"):
            audios = batch["audio"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _, loss = model(input_ids, audios, attention_mask=attention_mask, targets=labels)

            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_train_loss += batch_loss

            if global_step % 50 == 0:
                print(f"Stage3 Step: {global_step}, Instruction Loss: {batch_loss:.4f}")

            global_step += 1

        avg_train_loss = total_train_loss / len(train_loader)
        
        # 评估性能
        if train_cfg.eval_in_epochs:
            accuracy = test_savee(model, tokenizer, test_loader, device)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model.save_pretrained(save_directory=f"{alm_cfg.alm_checkpoint_path}/stage3_best")
            print(f"Stage3 Epoch {epoch+1}/{train_cfg.stage3_epochs} | Loss: {avg_train_loss:.4f} | Accuracy: {accuracy:.4f}")
        else:
            print(f"Stage3 Epoch {epoch+1}/{train_cfg.stage3_epochs} | Instruction Loss: {avg_train_loss:.4f}")

    # 保存最终模型
    model.save_pretrained(save_directory=f"{alm_cfg.alm_checkpoint_path}/final_model")
    print("Stage 3 completed!")
    return model

def train_three_stages(train_cfg, alm_cfg):
    """完整的三阶段训练"""
    print("Starting Three-Stage Training Pipeline")
    
    # 第一阶段：模态投影器对齐
    stage1_model = train_step1_alignment(train_cfg, alm_cfg)
    
    # 第二阶段：语言模型预训练
    stage2_model = train_step2_pretraining(train_cfg, alm_cfg, stage1_model)
    
    # 第三阶段：指令微调
    final_model = train_step3_instruction_tuning(train_cfg, alm_cfg, stage2_model)
    
    print("=== Training Pipeline Completed! ===")
    return final_model


@dataclass
class TrainConfig:
    lr_mp: float = 1e-4
    lr_backbones: float = 5e-6
    val_ratio: float = 0.2
    compile: bool = False
    data_cutoff_idx: int = 1024
    batch_size: int = 8  # 减小以适应三阶段训练
    savee_batch_size: int = 8
    
    # 三个阶段的epoch数
    stage1_epochs: int = 5   # 模态对齐
    stage2_epochs: int = 10  # 预训练 
    stage3_epochs: int = 5   # 指令微调
    
    eval_in_epochs: bool = False
    resume_from_alm_checkpoint: bool = False

    train_dataset_path: str = 'MLCommons/peoples_speech'
    train_dataset_name: tuple[str, ...] = ('clean_sa', )
    test_dataset_path: str = "AbstractTTS/SAVEE"


# 替换原来的训练调用
alm_cfg = ALMConfig()
train_cfg = TrainConfig()

# 运行三阶段训练
final_model = train_three_stages(train_cfg, alm_cfg)