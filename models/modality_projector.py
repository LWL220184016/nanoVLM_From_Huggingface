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

    def get_parameter_count(self):
        """返回当前投影器的参数统计"""
        return get_modality_projector_info(self)
    
    def print_stats(self):
        """打印当前投影器的统计信息"""
        print_projector_stats(self)

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

    def get_parameter_count(self):
        """返回当前投影器的参数统计"""
        return get_modality_projector_info(self)
    
    def print_stats(self):
        """打印当前投影器的统计信息"""
        print_projector_stats(self)

class HybridModalityProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.audio_hidden_dim
        self.output_dim = cfg.lm_hidden_dim
        self.target_length = cfg.mp_target_length
        
        # 新增参数配置
        self.conv_layers_num = getattr(cfg, 'mp_conv_layers', 3)  # 卷积层数量
        self.conv_kernel_size = getattr(cfg, 'mp_conv_kernel_size', 3)  # 卷积核大小
        self.conv_stride = getattr(cfg, 'mp_conv_stride', 1)  # 卷积步长
        self.conv_channels_multiplier = getattr(cfg, 'mp_conv_channels_multiplier', 2)  # 通道倍数
        self.use_residual = getattr(cfg, 'mp_use_residual', True)  # 是否使用残差连接
        self.dropout_rate = getattr(cfg, 'mp_dropout_rate', 0.1)  # dropout率
        self.transformer_layers = getattr(cfg, 'mp_transformer_layers', 2)  # Transformer层数
        self.transformer_heads = getattr(cfg, 'mp_transformer_heads', 8)  # 注意力头数
        self.use_layer_scale = getattr(cfg, 'mp_use_layer_scale', False)  # 是否使用LayerScale
        self.layer_scale_init = getattr(cfg, 'mp_layer_scale_init', 1e-6)  # LayerScale初始值
        
        # 动态构建卷积层
        self.conv_layers = self._build_conv_layers()
        
        # 注意力池化到目标长度
        self.attention_pool = AttentionPooling(
            self.output_dim, 
            self.target_length,
            num_heads=getattr(cfg, 'mp_attention_heads', 8),
            dropout=self.dropout_rate
        )
        
        # 动态构建Transformer层
        self.transformer = self._build_transformer_layers()
        
        # 输出投影（增加参数控制）
        self.output_hidden_multiplier = getattr(cfg, 'mp_output_hidden_multiplier', 2)
        self.output_projection = self._build_output_projection()
        
        # LayerScale支持
        if self.use_layer_scale:
            self.layer_scale = nn.Parameter(
                torch.ones(self.output_dim) * self.layer_scale_init
            )
        
        self.apply(self._init_weights)

    def _build_conv_layers(self):
        """动态构建卷积层"""
        layers = []
        in_channels = self.input_dim
        
        for i in range(self.conv_layers_num):
            if i == self.conv_layers_num - 1:
                # 最后一层输出到目标维度
                out_channels = self.output_dim
            else:
                out_channels = in_channels * self.conv_channels_multiplier
            
            # 卷积层
            conv_layer = nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size=self.conv_kernel_size, 
                stride=self.conv_stride,
                padding=self.conv_kernel_size // 2
            )
            layers.append(conv_layer)
            
            # 批归一化（除了最后一层）
            if i < self.conv_layers_num - 1:
                layers.append(nn.BatchNorm1d(out_channels))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(self.dropout_rate))
                
                # 残差连接（如果维度匹配）
                if self.use_residual and in_channels == out_channels:
                    # 这里会在forward中处理残差连接
                    pass
            
            in_channels = out_channels
        
        return nn.ModuleList(layers)
    
    def _build_transformer_layers(self):
        """动态构建Transformer层"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.output_dim,
            nhead=self.transformer_heads,
            dim_feedforward=self.output_dim * 4,
            dropout=self.dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=getattr(self.cfg, 'mp_norm_first', True)  # Pre-norm vs Post-norm
        )
        
        return nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.transformer_layers
        )
    
    def _build_output_projection(self):
        """动态构建输出投影层"""
        hidden_dim = self.output_dim * self.output_hidden_multiplier
        
        layers = [
            nn.Linear(self.output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim)
        ]
        
        # 可选的额外投影层
        if getattr(self.cfg, 'mp_extra_projection_layers', 0) > 0:
            extra_layers = []
            for _ in range(self.cfg.mp_extra_projection_layers):
                extra_layers.extend([
                    nn.Linear(self.output_dim, self.output_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout_rate)
                ])
            layers = layers[:-1] + extra_layers + [layers[-1]]  # 保持LayerNorm在最后
        
        return nn.Sequential(*layers)

    def _init_weights(self, module):
        """改进的权重初始化"""
        if isinstance(module, nn.Linear):
            # 使用不同的初始化策略
            init_method = getattr(self.cfg, 'mp_weight_init', 'xavier')
            if init_method == 'xavier':
                torch.nn.init.xavier_uniform_(module.weight)
            elif init_method == 'kaiming':
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif init_method == 'normal':
                torch.nn.init.normal_(module.weight, std=0.02)
            
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # x: [B, T, input_dim]
        batch_size = x.shape[0]
        
        # 卷积特征提取（带残差连接）
        x = x.transpose(1, 2)  # [B, input_dim, T]
        
        # 逐层应用卷积，支持残差连接
        conv_idx = 0
        for i in range(self.conv_layers_num):
            identity = x if self.use_residual else None
            
            # 卷积层
            x = self.conv_layers[conv_idx](x)
            conv_idx += 1
            
            # 中间层的激活和归一化
            if i < self.conv_layers_num - 1:
                # BatchNorm
                x = self.conv_layers[conv_idx](x)
                conv_idx += 1
                # GELU
                x = self.conv_layers[conv_idx](x)
                conv_idx += 1
                # Dropout
                x = self.conv_layers[conv_idx](x)
                conv_idx += 1
                
                # 残差连接（如果维度匹配）
                if (self.use_residual and identity is not None and 
                    identity.shape == x.shape):
                    x = x + identity
        
        x = x.transpose(1, 2)  # [B, T, output_dim]
        
        # 注意力池化
        x = self.attention_pool(x)  # [B, target_length, output_dim]
        
        # Transformer细化
        x = self.transformer(x)  # [B, target_length, output_dim]
        
        # LayerScale（如果启用）
        if self.use_layer_scale:
            x = x * self.layer_scale
        
        # 输出投影
        x = self.output_projection(x)
        
        return x

    def get_parameter_count(self):
        """返回当前投影器的参数统计"""
        return get_modality_projector_info(self)
    
    def print_stats(self):
        """打印当前投影器的统计信息"""
        print_projector_stats(self)
        
        # 打印超参数配置
        print(f"\n=== HybridModalityProjector 超参数配置 ===")
        print(f"卷积层数: {self.conv_layers_num}")
        print(f"卷积核大小: {self.conv_kernel_size}")
        print(f"通道倍数: {self.conv_channels_multiplier}")
        print(f"使用残差连接: {self.use_residual}")
        print(f"Dropout率: {self.dropout_rate}")
        print(f"Transformer层数: {self.transformer_layers}")
        print(f"注意力头数: {self.transformer_heads}")
        print(f"使用LayerScale: {self.use_layer_scale}")
        if self.use_layer_scale:
            print(f"LayerScale初始值: {self.layer_scale_init}")
        print("=" * 50)

class AttentionPooling(nn.Module):
    """改进的基于注意力的池化层"""
    def __init__(self, hidden_dim, target_length, num_heads=8, dropout=0.1):
        super().__init__()
        self.target_length = target_length
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 可学习的查询向量
        self.queries = nn.Parameter(torch.randn(target_length, hidden_dim))
        
        # 注意力计算
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
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
        
        # 层归一化
        pooled_features = self.layer_norm(pooled_features)
        
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

    def get_parameter_count(self):
        """返回当前投影器的参数统计"""
        return get_modality_projector_info(self)
    
    def print_stats(self):
        """打印当前投影器的统计信息"""
        print_projector_stats(self)

def count_parameters(model):
    """计算模型的总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_modality_projector_info(projector):
    """获取模态投影器的详细信息"""
    total_params, trainable_params = count_parameters(projector)
    
    info = {
        'type': projector.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'parameter_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
        'input_dim': getattr(projector, 'input_dim', 'N/A'),
        'output_dim': getattr(projector, 'output_dim', 'N/A'),
        'target_length': getattr(projector, 'target_length', 'N/A')
    }
    
    return info

def print_projector_stats(projector):
    """打印模态投影器的统计信息"""
    info = get_modality_projector_info(projector)
    
    print(f"\n=== {info['type']} 统计信息 ===")
    print(f"总参数量: {info['total_parameters']:,}")
    print(f"可训练参数量: {info['trainable_parameters']:,}")
    print(f"参数大小: {info['parameter_size_mb']:.2f} MB")
    print(f"输入维度: {info['input_dim']}")
    print(f"输出维度: {info['output_dim']}")
    print(f"目标长度: {info['target_length']}")
    
    # 计算参数分布
    if hasattr(projector, 'named_parameters'):
        print("\n层级参数分布:")
        layer_params = {}
        for name, param in projector.named_parameters():
            layer_name = name.split('.')[0]  # 获取顶层模块名
            if layer_name not in layer_params:
                layer_params[layer_name] = 0
            layer_params[layer_name] += param.numel()
        
        for layer_name, param_count in layer_params.items():
            percentage = param_count / info['total_parameters'] * 100
            print(f"  {layer_name}: {param_count:,} ({percentage:.1f}%)")
    
    print("=" * 50)


if __name__ == "__main__":
    # 示例配置
    from config import ALMConfig 
    
    cfg = ALMConfig()
    
    # 创建投影器实例
    projector = create_modality_projector(cfg)
    
    # 打印参数统计信息
    projector.print_stats()
    