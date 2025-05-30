import torch
from torch.utils.data import Dataset

import models.config as cfg

class AudioQADataset(Dataset): # https://huggingface.co/datasets/AbstractTTS/IEMOCAP
    def __init__(self, dataset, tokenizer, audio_processor):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 处理音频
        audio = item['audio']  # 假设数据集包含音频路径
        original_sampling_rate = item['sampling_rate']

        processed_audio = self.audio_processor(audio['array'], original_sampling_rate)
        
        
        return {
            "audio": processed_audio,
            "gender": item['gender'],
            "transcription": item['transcription'],
            "major_emotion": item['major_emotion']
        }
    
class SAVEEDataset(Dataset):  # https://huggingface.co/datasets/AbstractTTS/SAVEE
    def __init__(self, dataset, tokenizer, audio_processor):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        audio = item['audio']  # 假设数据集包含音频路径
        original_sampling_rate = item['sampling_rate']

        processed_audio = self.audio_processor(audio['array'], original_sampling_rate)
        transcription = item['transcription'] + self.tokenizer.eos_token # Add EOS token to the answer to train model to predict it, enabling correct stopping during generation
        
        return {
            "audio": processed_audio,
            "gender": item['gender'],
            "transcription": transcription,
            "major_emotion": item['emotion']
        }