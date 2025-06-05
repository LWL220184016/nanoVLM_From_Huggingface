import torch
from torch.utils.data import Dataset
import soundfile as sf
import io
import numpy as np
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
        if 'wav' in item and isinstance(item['wav'], dict):
            audio_feature_dict = item['wav']
        elif 'audio' in item and isinstance(item['audio'], dict):
            audio_feature_dict = item['audio'] # Get the dictionary containing audio data
        else:
            raise KeyError("Audio data not found in 'audio' or 'wav' keys, or format is incorrect.")
            
        raw_audio_array = None
        original_sampling_rate = None
        _sr_from_bytes_decode = None
        try: 
            raw_audio_array = audio_feature_dict['array']
            # Ensure it's a numpy array if it comes from 'array' key
            if not isinstance(raw_audio_array, np.ndarray):
                raw_audio_array = np.array(raw_audio_array)
        except KeyError: # 'array' key was not in audio_feature_dict, try 'bytes'.
            if 'bytes' in audio_feature_dict:
                audio_bytes = audio_feature_dict['bytes']
                try:
                    # Convert bytes to numpy array and get its sampling rate
                    data, sr_val = sf.read(io.BytesIO(audio_bytes))
                    raw_audio_array = data # sf.read returns a numpy array
                    _sr_from_bytes_decode = sr_val # Store sampling rate from decoded bytes
                except Exception as e:
                    raise ValueError(f"Failed to convert audio from 'bytes' key in audio_feature_dict: {e}")
            else:
                # Neither 'array' nor 'bytes' found in the pre-selected audio_feature_dict.
                raise KeyError("Neither 'array' nor 'bytes' found in audio_feature_dict for audio data.")
        
        if _sr_from_bytes_decode is not None:
            # If audio was decoded from bytes, use the sampling rate obtained from decoding
            original_sampling_rate = _sr_from_bytes_decode
        else:
            # Otherwise (e.g., 'array' was used, or 'bytes' processing failed before sr_val was set),
            # try to get sampling_rate from the audio_feature_dict or default.
            try: 
                original_sampling_rate = audio_feature_dict['sampling_rate']
            except KeyError:
                # This fallback is used if 'sampling_rate' is not in audio_feature_dict
                # and we didn't get it from decoding bytes.
                print(f"Warning: 'sampling_rate' not found in audio_feature_dict for item {idx} and not decoded from bytes. Defaulting to 16000.")
                original_sampling_rate = 16000

        processed_audio = self.audio_processor(raw_audio_array, original_sampling_rate)
        gender = item.get('gender', item.get('sex')) # Try 'gender', then 'sex'
        transcription = item.get('transcription', item.get('text')) # Try 'gender', then 'sex'
        
        return {
            "audio": processed_audio,
            "gender": gender,
            "transcription": transcription,
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
        
        audio_feature_dict = item['audio'] # Get the dictionary containing audio data
        raw_audio_array = audio_feature_dict['array']
        original_sampling_rate = audio_feature_dict['sampling_rate'] # <--- 修正點

        processed_audio = self.audio_processor(raw_audio_array, original_sampling_rate)
        transcription = item['transcription'] + self.tokenizer.eos_token # Add EOS token to the answer to train model to predict it, enabling correct stopping during generation
        gender = item.get('gender', item.get('sex')) # Try 'gender', then 'sex'
        
        return {
            "audio": processed_audio,
            "gender": gender,
            "transcription": transcription,
            "major_emotion": item['emotion']
        }