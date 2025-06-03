from dataclasses import dataclass

@dataclass
class ALMConfig:
    # 将原来的vit_相关配置改为audio_相关配置
    audio_hidden_dim: int = 768
    audio_inter_dim: int = 4 * audio_hidden_dim
    audio_patch_size: int = 16  # 音频patch大小（时间步数）
    audio_n_heads: int = 12
    audio_dropout: float = 0.0
    audio_n_blocks: int = 12
    audio_ln_eps: float = 1e-6
    audio_model_type: str = 'custom_audio_transformer'
    
    # 音频处理相关参数
    audio_sample_rate: int = 16000  # 采样率
    audio_n_fft: int = 400  # FFT窗口大小
    audio_hop_length: int = 160  # 跳跃长度
    audio_n_mels: int = 80  # 梅尔滤波器数量
    audio_max_length: int = 1000  # 最大时间步数

    lm_hidden_dim: int = 576
    lm_inter_dim: int = 1536
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_vocab_size: int = 49152
    lm_n_heads: int = 9
    lm_n_kv_heads: int = 3
    lm_dropout: float = 0.0
    lm_n_blocks: int = 30
    lm_attn_scaling: float = 1.0
    IMAGE_TOKEN_LENGTH: int = 49
    TOTAL_SEQUENCE_LENGTH: int = 128
    lm_max_length: int = TOTAL_SEQUENCE_LENGTH - IMAGE_TOKEN_LENGTH  # Maximum length for the language model, derived from TOTAL_SEQUENCE_LENGTH and IMAGE_TOKEN_LENGTH
    lm_use_tokens: bool = False # Decide if the LM expects tokens or embeddings as input (if using as a backbone for the VLM, set to False)
    lm_tie_weights: bool = True # Decide if you want to tie the LM Head weight to the token embedding weights
    lm_model_type: str = 'HuggingFaceTB/SmolLM2-135M'
    lm_tokenizer: str = 'HuggingFaceTB/cosmo2-tokenizer'
    lm_eos_token_id: int = 0

    # 模态投影器配置（音频专用）
    mp_projection_type: str = 'adaptive_pool'  # 'linear', 'adaptive_pool', 'attention', 'conv_downsample'
    mp_target_length: int = 50  # 目标音频token长度
    mp_use_position_aware: bool = True  # 是否使用位置感知
    audio_token_target_length: int = 50  # 最终的音频token数量
    
    # 更新lm_max_length
    lm_max_length: int = 128 - 50  # 为音频token预留50个位置

    # 音频语言模型配置
    alm_load_backbone_weights: bool = True
    alm_checkpoint_path: str = 'checkpoints/nanoALM-222M'
    
    # 音频token长度（替代原来的IMAGE_TOKEN_LENGTH）
    AUDIO_TOKEN_LENGTH: int = 62  # audio_max_length // audio_patch_size
    TOTAL_SEQUENCE_LENGTH: int = 128
    lm_max_length: int = TOTAL_SEQUENCE_LENGTH - AUDIO_TOKEN_LENGTH


@dataclass
class TrainConfig:
    lr_mp: float = 2e-3
    lr_backbones: float = 1e-4
    data_cutoff_idx: int = None
    val_ratio: float = 0.025
    batch_size: int = 256
    gradient_accumulation_steps: int = 1
    savee_batch_size: int = 32
    max_grad_norm: float = None
    eval_in_epochs: bool = True
    eval_interval: int = 250
    epochs: int = 5
    compile: bool = False
    resume_from_vlm_checkpoint: bool = False # Indicate if the training should be resumed from a checkpoint of the whole VLM or you want to start from scratch
    train_dataset_path: str = 'AbstractTTS/IEMOCAP'
    train_dataset_name: tuple[str, ...] = ('all_data', )
    test_dataset_path: str = "AbstractTTS/SAVEE"
    wandb_entity: str = "HuggingFace" # Indicate the entity to log to in wandb
    log_wandb: bool = True
