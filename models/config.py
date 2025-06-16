from dataclasses import dataclass

@dataclass
class ALMConfig:
    # 音頻編碼器配置
    audio_hidden_dim: int = 1280
    audio_inter_dim: int = 4 * audio_hidden_dim
    audio_patch_size: int = 16  # 音頻patch大小（時間步數）
    audio_n_heads: int = 12
    audio_dropout: float = 0.0
    audio_n_blocks: int = 12
    audio_ln_eps: float = 1e-6
    # audio_model_type: str = 'custom_audio_transformer'
    # 如果使用以下 ASR 模型, 以上參數將不會發生作用
    # audio_model_type: str = 'nvidia/parakeet-tdt-0.6b-v2' # asr model for encoder from NeMo
    audio_model_type: str = 'openai/whisper-large-v3' # asr model for encoder from huggingface

    # 音頻處理相關參數
    audio_sample_rate: int = 16000  # 採樣率
    audio_n_fft: int = 400  # FFT窗口大小
    audio_hop_length: int = 160  # 跳躍長度
    audio_n_mels: int = 80  # 梅爾濾波器數量
    audio_max_length: int = 3000  # 最大時間步數

    # 語言模型配置
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
    lm_use_tokens: bool = False
    lm_tie_weights: bool = True
    lm_model_type: str = 'HuggingFaceTB/SmolLM2-135M'
    lm_tokenizer: str = 'HuggingFaceTB/cosmo2-tokenizer'
    lm_eos_token_id: int = 0

    # 音頻token配置
    # AUDIO_TOKEN_LENGTH: int = int(audio_max_length / audio_patch_size)
    TOTAL_SEQUENCE_LENGTH: int = 128
    
    # 模態投影器配置
    mp_projection_type: str = 'adaptive_pool'
    mp_target_length: int = 50
    mp_use_position_aware: bool = True
    
    # 計算最大語言模型長度
    lm_max_length: int = TOTAL_SEQUENCE_LENGTH - mp_target_length

    # ALM配置
    alm_name: str = "nanoALM-222M"
    alm_load_backbone_weights: bool = True
    alm_checkpoint_path: str = 'checkpoints/nanoALM-222M'
    hf_repo_name: str = None  # 設置您的 HuggingFace 倉庫名稱


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
    resume_from_alm_checkpoint: bool = False
    # train_dataset_path: str = 'AbstractTTS/IEMOCAP' # for emotion recognition, disabled
    # train_dataset_name: tuple[str, ...] = ('default', ) # for emotion recognition, disabled
    train_dataset_path: str = 'speechbrain/LoquaciousSet'
    train_dataset_name: tuple[str, ...] = ('medium', )
    test_dataset_path: str = "AbstractTTS/SAVEE"
    wandb_entity: str = "HuggingFace"
    log_wandb: bool = True