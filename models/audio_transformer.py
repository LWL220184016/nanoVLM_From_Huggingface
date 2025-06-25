import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np

class AudioFeatureExtractor(nn.Module):
    """音频特征提取器，将音频转换为频谱特征"""
    def __init__(self, cfg):
        super().__init__()
        self.sample_rate = cfg.audio_sample_rate
        self.n_fft = cfg.audio_n_fft
        self.hop_length = cfg.audio_hop_length
        self.n_mels = cfg.audio_n_mels
        self.max_length = cfg.audio_max_length
        
    def forward(self, audio_batch):
        """
        audio_batch: [batch_size, audio_length] 原始音频波形
        返回: [batch_size, time_steps, n_mels] 梅尔频谱
        """
        batch_size = audio_batch.shape[0]
        mel_specs = []
        
        for i in range(batch_size):
            audio = audio_batch[i].cpu().numpy()
            # 计算梅尔频谱
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            # 转换为对数刻度
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            mel_specs.append(mel_spec)
        
        # 标准化并转换为张量
        mel_specs = np.stack(mel_specs)
        mel_specs = torch.tensor(mel_specs, dtype=torch.float32, device=audio_batch.device)
        
        # 转置维度：[batch_size, n_mels, time_steps] -> [batch_size, time_steps, n_mels]
        mel_specs = mel_specs.transpose(1, 2)
        
        # 截断或填充到固定长度
        if mel_specs.shape[1] > self.max_length:
            mel_specs = mel_specs[:, :self.max_length, :]
        else:
            pad_length = self.max_length - mel_specs.shape[1]
            mel_specs = F.pad(mel_specs, (0, 0, 0, pad_length))
            
        return mel_specs

class AudioPatchEmbeddings(nn.Module):
    """将音频频谱切分为patches并嵌入"""
    def __init__(self, cfg):
        super().__init__()
        self.patch_size = cfg.audio_patch_size
        self.hidden_dim = cfg.audio_hidden_dim
        self.n_mels = cfg.audio_n_mels
        
        # patch的维度
        self.patch_dim = self.patch_size * self.n_mels
        self.projection = nn.Linear(self.patch_dim, self.hidden_dim)
        
    def forward(self, x):
        """
        x: [batch_size, time_steps, n_mels]
        返回: [batch_size, num_patches, hidden_dim]
        """
        batch_size, time_steps, n_mels = x.shape
        
        # 计算patch数量
        num_patches = time_steps // self.patch_size
        
        # 截断到能被patch_size整除的长度
        x = x[:, :num_patches * self.patch_size, :]
        
        # 重塑为patches: [batch_size, num_patches, patch_size * n_mels]
        # Use reshape instead of view to handle non-contiguous tensors
        x = x.reshape(batch_size, num_patches, self.patch_size * n_mels)
        
        # 投影到隐藏维度
        x = self.projection(x)
        
        return x

class AudioMultiHeadAttention(nn.Module):
    """音频Transformer的多头注意力机制"""
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.audio_n_heads
        self.embd_dim = cfg.audio_hidden_dim
        assert self.embd_dim % self.n_heads == 0
        self.head_dim = self.embd_dim // self.n_heads
        self.dropout = cfg.audio_dropout
        
        self.qkv_proj = nn.Linear(self.embd_dim, 3 * self.embd_dim, bias=True)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=True)
        
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        
        self.sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        if self.sdpa:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        
        return y

class AudioMLP(nn.Module):
    """音频Transformer的前馈网络"""
    def __init__(self, cfg):
        super().__init__()
        self.hidden_dim = cfg.audio_hidden_dim
        self.inter_dim = cfg.audio_inter_dim
        
        self.fc1 = nn.Linear(self.hidden_dim, self.inter_dim)
        self.fc2 = nn.Linear(self.inter_dim, self.hidden_dim)
        self.dropout = nn.Dropout(cfg.audio_dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class AudioBlock(nn.Module):
    """音频Transformer块"""
    def __init__(self, cfg):
        super().__init__()
        self.attn = AudioMultiHeadAttention(cfg)
        self.mlp = AudioMLP(cfg)
        self.norm1 = nn.LayerNorm(cfg.audio_hidden_dim, eps=cfg.audio_ln_eps)
        self.norm2 = nn.LayerNorm(cfg.audio_hidden_dim, eps=cfg.audio_ln_eps)
        
    def forward(self, x):
        # Pre-norm架构
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class AudioTransformer(nn.Module):
    """完整的音频Transformer编码器"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 音频特征提取和patch嵌入
        self.feature_extractor = AudioFeatureExtractor(cfg)
        self.patch_embeddings = AudioPatchEmbeddings(cfg)
        
        # 位置编码
        max_patches = cfg.audio_max_length // cfg.audio_patch_size
        self.position_embeddings = nn.Parameter(torch.randn(1, max_patches, cfg.audio_hidden_dim))
        
        # Transformer块
        self.blocks = nn.ModuleList([
            AudioBlock(cfg) for _ in range(cfg.audio_n_blocks)
        ])
        
        # 最终层归一化
        self.norm = nn.LayerNorm(cfg.audio_hidden_dim, eps=cfg.audio_ln_eps)
        self.dropout = nn.Dropout(cfg.audio_dropout)
        
    def forward(self, audio):
        """
        audio: [batch_size, audio_length] 原始音频波形
        返回: [batch_size, num_patches, hidden_dim] 音频特征
        """
        # 提取音频特征
        x = self.feature_extractor(audio)  # [B, time_steps, n_mels]
        
        # 转换为patches
        x = self.patch_embeddings(x)  # [B, num_patches, hidden_dim]
        
        # 添加位置编码
        seq_len = x.shape[1]
        x = x + self.position_embeddings[:, :seq_len, :]
        x = self.dropout(x)
        
        # 通过Transformer块
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x
    
    @classmethod
    def from_pretrained(cls, cfg):
        """从预训练权重加载（如果有的话）"""
        # 这里可以实现从HuggingFace Hub加载预训练的音频模型
        # 比如Wav2Vec2, Whisper等的编码器部分
        print(f"Initializing AudioTransformer from scratch with {cfg.audio_model_type}")
        model = cls(cfg)
        
        # 初始化权重
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
        
        model.apply(init_weights)
        print(f"AudioTransformer initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")
        return model
    
class AudioTransformer_from_NeMo():
    """完整的音频Transformer编码器"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.asr_model = self.from_pretrained(cfg)
        

    def forward(self, audio):
        """
        audio: [batch_size, audio_length] 原始音频波形
        返回: [batch_size, num_patches, hidden_dim] 音频特征
        """
        with torch.no_grad():

            # NeMo preprocessors typically expect mono audio. If stereo, convert to mono.
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # NeMo expects a batch of audio, so add a batch dimension (batch_size, num_samples)
            input_signal = audio.squeeze(0).unsqueeze(0) # Shape: (1, num_samples)
            
            # Get the length of the audio in samples for the batch
            length = torch.tensor([input_signal.shape[1]], dtype=torch.long) # Shape: (1,)

            # Step 2: Call the preprocessor with the correctly formatted input_signal and length
            processed_signal, processed_signal_len = self.asr_model.preprocessor(
                input_signal=input_signal,  # This is the audio tensor
                length=length              # This is the length tensor
            )
            
            # Step 3: Call the encoder
            encoded, encoded_len = self.asr_model.encoder(
                audio_signal=processed_signal,
                length=processed_signal_len
            )
        return encoded

    @classmethod
    def from_pretrained(cls, cfg):
        """从预训练权重加载（如果有的话）"""
        # 这里可以实现从HuggingFace Hub加载预训练的音频模型
        # 比如Wav2Vec2, Whisper等的编码器部分
        import nemo.collections.asr as nemo_asr
        asr_model = nemo_asr.models.ASRModel.from_pretrained(cfg.audio_model_type)

        return asr_model
    
class AudioTransformer_from_HF():
    """完整的音频Transformer编码器"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.asr_model = self.from_pretrained(cfg)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.asr_model.to(self.device)
        self.asr_model.eval()
        self.datatype = torch.float32
        

    def forward(self, audio, output_hidden_states=True):
        """
        audio: [batch_size, audio_length] 原始音频波形
        返回: [batch_size, num_patches, hidden_dim] 音频特征
        """

        # processer 和 ASR 模型的 processor 一樣, 這裡被註解是因為是先前已經經過 processor處理了
        # audio = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = audio.input_features.to(self.device, dtype=self.datatype)

        # 產生音訊編碼 (encoder embeddings)
        with torch.no_grad():
            # 直接呼叫模型的 encoder
            encoder_outputs = self.asr_model.model.encoder(input_features, output_hidden_states=True)

        # 提取最後一層的隱藏狀態
        # 這就是音訊的編碼/嵌入
        encoded = encoder_outputs.last_hidden_state
        return encoded

    @classmethod
    def from_pretrained(cls, cfg):
        """从预训练权重加载（如果有的话）"""
        # 这里可以实现从HuggingFace Hub加载预训练的音频模型
        # 比如Wav2Vec2, Whisper等的编码器部分
        from transformers import WhisperModel

        # processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3") # moved to AudioProcessor in porcessors.py
        asr_model = WhisperModel.from_pretrained(cfg.audio_model_type)

        return asr_model