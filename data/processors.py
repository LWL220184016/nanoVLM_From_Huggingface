import torch
import numpy as np
from typing import Union, List
from transformers import AutoTokenizer, AutoProcessor

TOKENIZERS_CACHE = {}

def get_tokenizer(name):
    if name not in TOKENIZERS_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        TOKENIZERS_CACHE[name] = tokenizer
    return TOKENIZERS_CACHE[name]

def get_audio_processor(cfg):
    """获取音频处理器"""
    return AudioProcessor_from_HF(cfg)

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, max_length: float = 10.0):
        """
        音频处理器
        Args:
            sample_rate: 目标采样率
            max_length: 最大音频长度（秒）
        """
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.max_samples = int(sample_rate * max_length)
    
    def __call__(self, audio_array: str, input_sr) -> torch.Tensor:
        """
        处理音频文件
        Args:
            audio_path: 音频文件路径
        Returns:
            torch.Tensor: 处理后的音频张量 [max_samples]
        """
        # 加载音频
        audio = librosa.resample(y=audio_array, orig_sr=input_sr, target_sr=self.sample_rate)
        
        # 截断或填充到固定长度
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        else:
            padding = self.max_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
        
        return torch.tensor(audio, dtype=torch.float32)
    
    def batch_process(self, audio_paths: List[str]) -> torch.Tensor:
        """批处理音频文件"""
        batch_audio = []
        for path in audio_paths:
            audio = self(path)
            batch_audio.append(audio)
        return torch.stack(batch_audio)
    
class AudioProcessor_from_HF:
    def __init__(self, cfg):
        """
        音频处理器，使用 Hugging Face Transformers 的 processor。
        Args:
            cfg: 配置对象，应包含 audio_model_type 字符串，
                 例如 "openai/whisper-base" 或 "nvidia/parakeet-tdt-0.6b-v2"。
        """
        from transformers import AutoProcessor # 导入 AutoProcessor
        import numpy as np # 确保导入 numpy

        # 使用 AutoProcessor 根据 cfg.audio_model_type 自动加载合适的 processor。
        # 这能确保你为特定模型获取正确的预处理器。
        self.processor = AutoProcessor.from_pretrained(cfg.audio_model_type)
        self.audio_max_length = cfg.audio_max_length
        print(f"AudioProcessor initialized with model: {cfg.audio_model_type}, max_length: {self.audio_max_length}")
        # 大多数现代音频模型（包括Whisper和许多ASR模型）期望16kHz的采样率。
        # AutoProcessor 加载的 feature_extractor 通常会处理到目标采样率的重采样。

    def __call__(self, audio_array: np.ndarray, input_sr: int) -> torch.Tensor:
        """
        处理单个音频波形数据。
        Args:
            audio_array: 原始音频波形 (numpy array)。librosa.load 默认返回 float32 类型。
                         期望是单声道音频。
            input_sr: 原始音频的采样率。
        Returns:
            torch.Tensor: 模型所需的音频特征张量。
        """
        # 将音频数据和原始采样率传递给 Hugging Face processor。
        # processor 会负责：
        # 1. 重采样到模型期望的采样率（如果 input_sr 与模型期望不同）。
        # 2. 转换为模型所需的特征（例如，对数梅尔频谱）。
        # 3. 进行填充或截断以匹配模型的固定输入大小。
        # `padding="longest"` (或 `padding=True`) 会将音频（在批处理时）填充到批中最长序列的长度，
        # 或（对于单个项目，如Whisper）填充到模型的固定窗口大小。
        # `return_tensors="pt"` 会返回 PyTorch 张量。
        
        inputs = self.processor(
            audio_array, 
            sampling_rate=input_sr, 
            return_tensors="pt", 
        )
        
        # Processor 通常返回一个字典，其中包含 'input_features' (对于Whisper等)
        # 或 'input_values' (对于某些其他模型)。
        if hasattr(inputs, "input_features"):
            processed_audio = inputs.input_features
        elif hasattr(inputs, "input_values"):
            processed_audio = inputs.input_values
        else:
            # 你可能需要检查 self.processor 返回的具体键名
            raise ValueError(f"无法在处理器 {type(self.processor).__name__} 的输出中找到 'input_features' 或 'input_values'。")

        # 对于单个样本，处理器可能会返回 [1, num_features, sequence_length] 的形状。
        # 我们通常希望在后续通过 torch.stack 进行批处理时，每个样本是 [num_features, sequence_length]，
        # 因此，如果确实是单个样本且存在批次维度，则移除它。
        if processed_audio.ndim == 3 and processed_audio.shape[0] == 1:
            processed_audio = processed_audio.squeeze(0)
        
        print(f"Processed audio shape: {processed_audio.shape}")  # 调试输出
        return processed_audio
    
    def batch_process(self, audio_paths: List[str]) -> torch.Tensor:
        """
        批处理音频文件。
        Args:
            audio_paths: 音频文件路径列表。
        Returns:
            torch.Tensor: 批处理后的音频特征张量。
        """
        import librosa # 确保导入 librosa
        batch_features = []
        for path in audio_paths:
            # 从文件加载音频。sr=None 加载原始采样率，mono=True 确保单声道。
            # librosa.load 默认返回一个 np.ndarray (float32 类型) 和采样率。
            audio_array, input_sr = librosa.load(path, sr=None, mono=True)
            
            # 调用 __call__ 方法处理单个音频
            features = self(audio_array, input_sr)
            batch_features.append(features)
        
        # 使用 torch.stack 将单个特征张量列表堆叠成一个批处理张量。
        # 这要求所有特征张量具有相同的形状（通常由处理器中的填充策略保证）。
        return torch.stack(batch_features)