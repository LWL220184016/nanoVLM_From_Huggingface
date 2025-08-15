import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np

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
    def __init__(self, cfg, load_from_HF):
        from transformers import AutoConfig, AutoModel
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device

        if load_from_HF:
            print(f"Loading audio encoder from Huggingface: {cfg.audio_model_type}")
            self.audio_encoder = AutoModel.from_pretrained(cfg.audio_model_type).encoder
        else:
            print(f"Initializing empty audio encoder: {cfg.audio_model_type}")
            config = AutoConfig.from_pretrained(cfg.audio_model_type)
            self.audio_encoder = AutoModel.from_config(config).encoder

        self.audio_encoder.eval()
        self.audio_encoder.to(self.device)

    def forward(self, audio, output_hidden_states=True):
        """
        audio: [batch_size, audio_length] 原始音频波形
        返回: [batch_size, num_patches, hidden_dim] 音频特征
        """

        # processer 和 ASR 模型的 processor 一樣, 這裡被註解是因為是先前已經經過 processor處理了
        # audio = processor(audio, sampling_rate=16000, return_tensors="pt")

        # 產生音訊編碼 (encoder embeddings)
        with torch.no_grad():
            # 直接呼叫模型的 encoder
            # print(f"Debug(AudioTransformer_from_HF): audio.dtype: {audio.dtype}")
            encoder_outputs = self.audio_encoder(audio, output_hidden_states=output_hidden_states)

        # 提取最後一層的隱藏狀態
        # 這就是音訊的編碼/嵌入
        encoded = encoder_outputs.last_hidden_state
        return encoded

    # @classmethod
    # def from_pretrained(cls, cfg):
    #     """从预训练权重加载（如果有的话）"""
    #     # 这里可以实现从HuggingFace Hub加载预训练的音频模型
    #     # 比如Wav2Vec2, Whisper等的编码器部分
    #     from transformers import WhisperModel

    #     # processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3") # moved to AudioProcessor in porcessors.py
    #     asr_model = WhisperModel.from_pretrained(cfg.audio_model_type)

    #     return asr_model.encoder