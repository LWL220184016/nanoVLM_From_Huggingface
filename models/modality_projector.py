# Modality Projection from Audio to Language
import torch
import torch.nn as nn
import torch.nn.functional as F

# 在 audio_language_model.py 中
def create_modality_projector(cfg):
    """根据配置创建合适的模态投影器"""
    if cfg.mp_projection_type == 'adaptive':
        return ModalityProjector(cfg)
    elif cfg.mp_projection_type == 'transformer':
        return TransformerModalityProjector(cfg)
    elif cfg.mp_projection_type == 'hybrid':
        return HybridModalityProjector(cfg)

class ModalityProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.audio_hidden_dim  # 768 (Whisper)
        self.output_dim = cfg.lm_hidden_dim    # 576 (SmolLM2)
        self.target_length = cfg.mp_target_length  # 25
        
        # 增加中间维度和层数
        self.hidden_dim = self.input_dim * 2  # 1536
        
        # 多层特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        
        # 自适应池化（保持原有功能）
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.target_length)
        
        # 输出投影（多层）
        self.output_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(self.hidden_dim // 2, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        
        # 特征提取
        x = self.feature_extractor(x)  # [B, T, hidden_dim]
        
        # 自适应池化到目标长度
        x = x.transpose(1, 2)  # [B, hidden_dim, T]
        x = self.adaptive_pool(x)  # [B, hidden_dim, target_length]
        x = x.transpose(1, 2)  # [B, target_length, hidden_dim]
        
        # 输出投影
        x = self.output_projector(x)  # [B, target_length, output_dim]
        
        return x

class TransformerModalityProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.audio_hidden_dim
        self.output_dim = cfg.lm_hidden_dim
        self.target_length = cfg.mp_target_length
        
        # 输入投影
        self.input_projection = nn.Linear(self.input_dim, self.output_dim)
        
        # 可学习的查询向量（目标长度）
        self.queries = nn.Parameter(torch.randn(self.target_length, self.output_dim))
        
        # 多层Transformer
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.output_dim,
                nhead=8,
                dim_feedforward=self.output_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(4)  # 4层Transformer
        ])
        
        # 位置编码
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.target_length, self.output_dim)
        )
        
        # 输出层归一化
        self.output_norm = nn.LayerNorm(self.output_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 输入投影
        memory = self.input_projection(x)  # [B, T, output_dim]
        
        # 准备查询向量
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, target_length, output_dim]
        queries = queries + self.positional_encoding
        
        # 通过Transformer层
        output = queries
        for layer in self.transformer_layers:
            output = layer(output, memory)  # Cross-attention
        
        # 输出归一化
        output = self.output_norm(output)
        
        return output

class HybridModalityProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.audio_hidden_dim
        self.output_dim = cfg.lm_hidden_dim
        self.target_length = cfg.mp_target_length
        
        # 卷积特征提取（捕捉局部时序信息）
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.input_dim * 2),
            nn.GELU(),
            nn.Conv1d(self.input_dim * 2, self.input_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.input_dim * 2),
            nn.GELU(),
            nn.Conv1d(self.input_dim * 2, self.output_dim, kernel_size=1),
        )
        
        # 注意力池化到目标长度
        self.attention_pool = AttentionPooling(self.output_dim, self.target_length)
        
        # Transformer细化层
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.output_dim,
                nhead=8,
                dim_feedforward=self.output_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.LayerNorm(self.output_dim)
        )
        
        self.apply(self._init_weights)
    
    def forward(self, x):
        # x: [B, T, input_dim]
        
        # 卷积特征提取
        x = x.transpose(1, 2)  # [B, input_dim, T]
        x = self.conv_layers(x)  # [B, output_dim, T]
        x = x.transpose(1, 2)  # [B, T, output_dim]
        
        # 注意力池化
        x = self.attention_pool(x)  # [B, target_length, output_dim]
        
        # Transformer细化
        x = self.transformer(x)  # [B, target_length, output_dim]
        
        # 输出投影
        x = self.output_projection(x)
        
        return x

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim, target_length):
        super().__init__()
        self.target_length = target_length
        self.hidden_dim = hidden_dim
        
        # 可学习的查询向量
        self.queries = nn.Parameter(torch.randn(target_length, hidden_dim))
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 准备查询
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 注意力池化
        output, _ = self.attention(queries, x, x)
        
        return output

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