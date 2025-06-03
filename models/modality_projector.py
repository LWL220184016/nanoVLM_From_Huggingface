# Modality Projection from Audio to Language
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.audio_hidden_dim
        self.output_dim = cfg.lm_hidden_dim
        
        # 音频特定的投影策略
        self.projection_type = getattr(cfg, 'mp_projection_type', 'adaptive_pool')  # 'linear', 'adaptive_pool', 'attention'
        self.target_length = getattr(cfg, 'mp_target_length', 50)  # 目标序列长度
        
        if self.projection_type == 'linear':
            # 简单线性投影
            self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
            
        elif self.projection_type == 'adaptive_pool':
            # 自适应池化 + 线性投影
            self.adaptive_pool = nn.AdaptiveAvgPool1d(self.target_length)
            self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
            
        elif self.projection_type == 'attention':
            # 基于注意力的池化
            self.attention_pool = AttentionPooling(self.input_dim, self.target_length)
            self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
            
        elif self.projection_type == 'conv_downsample':
            # 卷积下采样
            self.conv_layers = nn.Sequential(
                nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )
            self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        x: [batch_size, seq_len, hidden_dim] 音频特征
        返回: [batch_size, target_length, lm_hidden_dim]
        """
        if self.projection_type == 'linear':
            # 直接线性投影，保持序列长度
            x = self.proj(x)
            
        elif self.projection_type == 'adaptive_pool':
            # 自适应池化到目标长度
            # x: [B, seq_len, hidden_dim] -> [B, hidden_dim, seq_len]
            x = x.transpose(1, 2)
            # 池化: [B, hidden_dim, target_length]
            x = self.adaptive_pool(x)
            # 转回: [B, target_length, hidden_dim]
            x = x.transpose(1, 2)
            # 线性投影
            x = self.proj(x)
            
        elif self.projection_type == 'attention':
            # 注意力池化
            x = self.attention_pool(x)
            x = self.proj(x)
            
        elif self.projection_type == 'conv_downsample':
            # 卷积下采样
            # x: [B, seq_len, hidden_dim] -> [B, hidden_dim, seq_len]
            x = x.transpose(1, 2)
            # 卷积下采样
            x = self.conv_layers(x)
            # 转回: [B, new_seq_len, hidden_dim]
            x = x.transpose(1, 2)
            # 线性投影
            x = self.proj(x)

        return x

class AttentionPooling(nn.Module):
    """基于注意力的池化层，将变长序列池化到固定长度"""
    def __init__(self, hidden_dim, target_length):
        super().__init__()
        self.target_length = target_length
        self.hidden_dim = hidden_dim
        
        # 可学习的查询向量
        self.queries = nn.Parameter(torch.randn(target_length, hidden_dim))
        
        # 注意力计算
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, hidden_dim]
        返回: [batch_size, target_length, hidden_dim]
        """
        batch_size = x.shape[0]
        
        # 扩展查询向量到batch
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 自注意力池化：queries作为Q，x作为K和V
        pooled_features, _ = self.attention(queries, x, x)
        
        return pooled_features

class AudioModalityProjector(nn.Module):
    """专门为音频设计的模态投影器"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.audio_hidden_dim
        self.output_dim = cfg.lm_hidden_dim
        
        # 计算原始audio token数量
        time_patches = cfg.audio_max_length // cfg.audio_patch_size
        freq_patches = cfg.audio_n_mels // cfg.audio_freq_patch_size
        self.original_length = time_patches * freq_patches
        
        # 目标长度（语言模型能处理的audio token数量）
        self.target_length = getattr(cfg, 'audio_token_target_length', 50)
        
        # 多层感知机进行维度和长度的双重投影
        self.dimension_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.input_dim // 2, self.output_dim)
        )
        
        # 序列长度自适应池化
        self.length_adapter = nn.AdaptiveAvgPool1d(self.target_length)
        
        # 位置感知的注意力池化（可选）
        self.use_position_aware = getattr(cfg, 'mp_use_position_aware', True)
        if self.use_position_aware:
            self.pos_encoding = nn.Parameter(torch.randn(1, self.original_length, self.input_dim) * 0.02)
            
    def forward(self, x):
        """
        x: [batch_size, seq_len, audio_hidden_dim] 
        返回: [batch_size, target_length, lm_hidden_dim]
        """
        # 添加位置编码
        if self.use_position_aware and hasattr(self, 'pos_encoding'):
            seq_len = min(x.shape[1], self.pos_encoding.shape[1])
            x[:, :seq_len, :] += self.pos_encoding[:, :seq_len, :]
        
        # 先进行维度投影
        x = self.dimension_proj(x)  # [B, seq_len, lm_hidden_dim]
        
        # 再进行长度自适应
        # 转置进行1D池化: [B, seq_len, dim] -> [B, dim, seq_len]
        x = x.transpose(1, 2)
        # 池化到目标长度: [B, dim, target_length]
        x = self.length_adapter(x)
        # 转回: [B, target_length, dim]
        x = x.transpose(1, 2)
        
        return x