import librosa
import torch
import numpy as np
from typing import Union, List

def get_audio_processor(sample_rate: int = 16000, max_length: float = 10.0):
    """获取音频处理器"""
    return AudioProcessor(sample_rate=sample_rate, max_length=max_length)

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
    
    def __call__(self, audio_path: str) -> torch.Tensor:
        """
        处理音频文件
        Args:
            audio_path: 音频文件路径
        Returns:
            torch.Tensor: 处理后的音频张量 [max_samples]
        """
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
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