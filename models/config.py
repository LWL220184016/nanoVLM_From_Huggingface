from dataclasses import dataclass

@dataclass
class ALMConfig:
    audio_hidden_dim: int = 768
    audio_inter_dim: int = 4 * audio_hidden_dim
    audio_patch_size: int = 16  # 音频patch大小（时间步数）
    audio_n_heads: int = 12
    audio_dropout: float = 0.0
    audio_n_blocks: int = 12
    audio_ln_eps: float = 1e-6
    # audio_model_type: str = 'custom_audio_transformer'
    # 如果使用 nvidia/parakeet-tdt-0.6b-v2, 以上參數將不會發生作用
    # audio_model_type: str = 'nvidia/parakeet-tdt-0.6b-v2' # asr model for encoder from huggingface
    # audio_model_type: str = 'openai/whisper-large-v3' # asr model for encoder from huggingface
    audio_model_type: str = 'openai/whisper-small.en' # asr model for encoder from huggingface

    # 音频处理相关参数
    audio_sample_rate: int = 16000  # 采样率
    audio_n_fft: int = 400  # FFT窗口大小
    audio_hop_length: int = 160  # 跳跃长度
    audio_n_mels: int = 80  # 梅尔滤波器数量
    audio_max_length: int = 1500  # 最大时间步数

    # ========== 语言模型配置选项 ==========
    # 选项1: SmolLM2-135M (当前使用，较小)
    # lm_hidden_dim: int = 576
    # lm_inter_dim: int = 1536
    # lm_vocab_size: int = 49152
    # lm_n_heads: int = 9
    # lm_n_kv_heads: int = 3
    # lm_n_blocks: int = 30
    # lm_model_type: str = 'HuggingFaceTB/SmolLM2-135M'
    # lm_tokenizer: str = 'HuggingFaceTB/cosmo2-tokenizer'
    
    # 选项2: SmolLM2-360M (中等大小)
    # lm_hidden_dim: int = 960
    # lm_inter_dim: int = 2560  
    # lm_vocab_size: int = 49152
    # lm_n_heads: int = 15
    # lm_n_kv_heads: int = 5
    # lm_n_blocks: int = 32
    # lm_model_type: str = 'HuggingFaceTB/SmolLM2-360M'
    # lm_tokenizer: str = 'HuggingFaceTB/cosmo2-tokenizer'
    
    # 选项3: SmolLM2-1.7B (较大)
    lm_hidden_dim: int = 2048
    lm_inter_dim: int = 5504
    lm_vocab_size: int = 49152
    lm_n_heads: int = 32
    lm_n_kv_heads: int = 32
    lm_n_blocks: int = 24
    lm_model_type: str = 'HuggingFaceTB/SmolLM2-1.7B'
    lm_tokenizer: str = 'HuggingFaceTB/cosmo2-tokenizer'
    
    # 选项4: Qwen2.5-0.5B (替代选择)
    # lm_hidden_dim: int = 896
    # lm_inter_dim: int = 4864
    # lm_vocab_size: int = 151936
    # lm_n_heads: int = 14
    # lm_n_kv_heads: int = 2
    # lm_n_blocks: int = 24
    # lm_model_type: str = 'Qwen/Qwen2.5-0.5B'
    # lm_tokenizer: str = 'Qwen/Qwen2.5-0.5B'
    
    # 选项5: Qwen2.5-1.5B (更大选择)
    # lm_hidden_dim: int = 1536
    # lm_inter_dim: int = 8960
    # lm_vocab_size: int = 151936
    # lm_n_heads: int = 12
    # lm_n_kv_heads: int = 2
    # lm_n_blocks: int = 28
    # lm_model_type: str = 'Qwen/Qwen2.5-1.5B'
    # lm_tokenizer: str = 'Qwen/Qwen2.5-1.5B'

    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_dropout: float = 0.0
    lm_attn_scaling: float = 1.0
    lm_eos_token_id: int = 0
    lm_use_tokens: bool = False
    lm_tie_weights: bool = True

    # 模態投影器配置
    mp_projection_type: str = 'default' # adaptive, transformer, hybrid, default
    mp_target_length: int = 25
    mp_use_position_aware: bool = True

    # 計算語言模型最大長度
    lm_max_length: int = 128 - mp_target_length  # 總長度 - 音頻token長度

    # ALM特定配置
    alm_load_backbone_weights: bool = True
    alm_checkpoint_path: str = 'checkpoints'
    alm_name: str = 'nanoALM-1.7B'  # 更新模型名称
    mp_hidden_multiplier = 2
    mp_dropout = 0.1


@dataclass
class TrainConfig:
    mp_pretrain_epochs: int = 3
    mp_pretrain_lr: float = 1e-4  # 预训练使用较高学习率

    lr_mp: float = 5e-6
    lr_backbones: float = 1e-6
    val_ratio: float = 0.2
    compile: bool = False
    data_cutoff_idx: int = 1024 # Let's only use a small subset of the data at first, otherwise it takes very long to see anything :D
    batch_size: int =  12 
    savee_batch_size: int = 8

    # epochs: int = 20
    stage1_epochs: int = 30  
    stage2_epochs: int = 12  
    stage3_epochs: int = 8  

    eval_in_epochs: bool = False # Deactivating this in colab, because it would evaluate 1500 samples of SAVEE every time otherwise
    resume_from_alm_checkpoint: bool = False # Indicate if the training should be resumed from a checkpoint of the whole ALM or you want to start from scratch

    # EOS token强化训练
    force_eos_training: bool = True
    eos_loss_weight: float = 2.0  # EOS token损失权重

    # train_dataset_path: str = 'AbstractTTS/IEMOCAP'
    # train_dataset_name: tuple[str, ...] = ('default', ) #All options; ("ai2d", "aokvqa", "chart2text", "chartqa", "clevr", "cocoqa", "datikz", "diagram_image_to_text", "docvqa", "dvqa", "figureqa", "finqa", "geomverse", "hateful_memes", "hitab", "iam", "iconqa", "infographic_vqa", "intergps", "localized_narratives", "mapqa", "multihiertt", "ocrvqa", "plotqa", "raven", "rendered_text", "robut_sqa", "robut_wikisql", "robut_wtq", "scienceqa", "screen2words", "st_vqa", "tabmwp", "tallyqa", "tat_qa", "textcaps", "textvqa", "tqa", "vistext", "visual7w", "visualmrc", "vqarad", "vqav2", "vsr", "websight") # "clevr_math", "okvqa", "spot_the_diff", "nlvr2", "mimic_cgd",

    # train_dataset_path: str = 'speechbrain/LoquaciousSet'
    # train_dataset_name: tuple[str, ...] = ('medium', ) # small, medium

    train_dataset_path: str = 'MLCommons/peoples_speech'
    train_dataset_name: tuple[str, ...] = ('clean_sa', ) # small, medium
    test_dataset_path: str = "AbstractTTS/SAVEE"
