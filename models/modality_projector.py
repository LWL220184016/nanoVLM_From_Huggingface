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
    