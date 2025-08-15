import torch
import numpy as np
from typing import Union, List
from transformers import AutoTokenizer, AutoProcessor

TOKENIZERS_CACHE = {}

def get_tokenizer(name):
    # TODO
    # 完成加載分詞器後保存分詞器, 如果檢查點有分詞器,優先加載檢查點的分詞器
    if name not in TOKENIZERS_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        if tokenizer.pad_token is None:
            # 常見做法：用 eos 當 pad
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        # 添加特殊 token 並調整嵌入層
        special_tokens_dict = {'additional_special_tokens': ['<AUDIO>']}
        tokenizer.add_special_tokens(special_tokens_dict)
        TOKENIZERS_CACHE[name] = tokenizer
    return TOKENIZERS_CACHE[name]

def get_audio_processor(cfg):
    """获取音频处理器"""
    return AudioProcessor_from_HF(cfg)

class AudioProcessor_from_HF:
    def __init__(self, cfg):
        """
        音频处理器，使用 Hugging Face Transformers 的 processor。
        Args:
            cfg: 配置对象，应包含 audio_model_type 字符串，
                 例如 "openai/whisper-base" 或 "nvidia/parakeet-tdt-0.6b-v2"。
                 Also, audio_max_length (for feature frames).
        """
        from transformers import AutoProcessor # 导入 AutoProcessor
        import numpy as np # 确保导入 numpy

        self.processor = AutoProcessor.from_pretrained(cfg.audio_model_type)
        self.target_feature_frames = cfg.audio_max_length  # Desired number of feature frames
        self.dtype = cfg.dtype

        # Get parameters from the loaded feature extractor to ensure consistency
        if hasattr(self.processor, 'feature_extractor') and \
           hasattr(self.processor.feature_extractor, 'hop_length') and \
           hasattr(self.processor.feature_extractor, 'n_fft') and \
           hasattr(self.processor.feature_extractor, 'sampling_rate'):
            
            self.hop_length = self.processor.feature_extractor.hop_length
            self.n_fft = self.processor.feature_extractor.n_fft
            # The processor will resample to this sampling_rate
            self.model_sampling_rate = self.processor.feature_extractor.sampling_rate
        else:
            # Fallback to cfg if attributes are not found, though less ideal
            # This might happen if the processor doesn't expose feature_extractor directly
            # or if it's a different type of processor.
            # For Whisper, the attributes should be available.
            print("Warning: Could not get hop_length, n_fft, sampling_rate from processor.feature_extractor. Using values from cfg.")
            self.hop_length = cfg.audio_hop_length
            self.n_fft = cfg.audio_n_fft
            self.model_sampling_rate = cfg.audio_sample_rate


        # Calculate the maximum number of raw audio samples required to produce target_feature_frames.
        # Formula for number of frames: n_frames = floor((n_samples - n_fft) / hop_length) + 1
        # So, to get n_frames, n_samples should be approximately (n_frames - 1) * hop_length + n_fft
        self.max_raw_audio_samples_for_target_frames = (self.target_feature_frames - 1) * self.hop_length + self.n_fft
        
        # Ensure audio_max_length from cfg is not misinterpreted as seconds here.
        # self.audio_max_length = cfg.audio_max_length # This was ambiguous, replaced by target_feature_frames

        print(f"AudioProcessor_from_HF initialized with model: {type(self.processor)}")
        print(f"  Target feature frames from cfg: {self.target_feature_frames}")
        print(f"  Using model sampling rate: {self.model_sampling_rate}, hop_length: {self.hop_length}, n_fft: {self.n_fft}")
        print(f"  Calculated max raw audio samples for processor: {self.max_raw_audio_samples_for_target_frames}")

    def __call__(self, audio_array: np.ndarray, input_sr: int) -> torch.Tensor:
        # 在處理前截斷音頻以減少記憶體使用
        max_samples = 30 * input_sr  # 限制為 30 秒
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]
        
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
        
        processed_audio = processed_audio.to(self.dtype)
        # print(f"Debug: Processed audio shape: {processed_audio.shape}")  # 调试输出
        del inputs
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
            audio_array, input_sr = librosa.load(path, sr=None, mono=True, dtype=np.float16)
            # print(f"Debug(AudioProcessor_from_HF): audio_array.dtype: {audio_array.dtype}")
            
            # 调用 __call__ 方法处理单个音频
            features = self(audio_array, input_sr)
            batch_features.append(features)
        
        # 使用 torch.stack 将单个特征张量列表堆叠成一个批处理张量。
        # 这要求所有特征张量具有相同的形状（通常由处理器中的填充策略保证）。
        return torch.stack(batch_features)