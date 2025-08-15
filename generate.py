import argparse
import torch
import librosa
import numpy as np
from PIL import Image

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from models.audio_language_model import AudioLanguageModel
from data.processors import get_tokenizer, get_audio_processor
from models.config import ALMConfig

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from an audio with nanoVLM")
    parser.add_argument(
        "--checkpoint", type=str, default="./3/",
        help="Path to a local checkpoint (directory or safetensors/pth). If omitted, we pull from HF."
    )
    parser.add_argument(
        "--hf_model", type=str, default="lusxvr/nanoVLM-222M",
        help="HuggingFace repo ID to download from incase --checkpoint isnt set."
    )
    parser.add_argument("--audio", type=str, default="./output_txt3.wav",
                        help="Path to input audio")
    parser.add_argument("--prompt", type=str, default="What is said in this audio?",
                        help="Text prompt to feed the model")
    parser.add_argument("--generations", type=int, default=5,
                        help="Num. of outputs to generate")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum number of tokens per output")
    return parser.parse_args()


def main():
    args = parse_args()

    device = ALMConfig.device
    print(f"Using device: {device}")

    source = args.checkpoint if args.checkpoint else args.hf_model
    print(f"Loading weights from: {source}")
    model = AudioLanguageModel.from_pretrained(source).to(device)
    model.eval()

    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    audio_processor = get_audio_processor(model.cfg)

    template = f"Question: {args.prompt} Answer:"
    encoded = tokenizer.batch_encode_plus([template], return_tensors="pt")
    tokens = encoded["input_ids"].to(device)

    # 使用 librosa 加載音頻文件來避免 torchaudio 兼容性問題
    try:
        audio_array, sr = librosa.load(args.audio, sr=16000)
        # 轉換為 torch tensor 並添加 batch 維度
        audio_tensor = torch.tensor(audio_array, dtype=torch.float16)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # 添加 channel 維度
        
        # 使用 audio_processor 處理音頻
        audio_t = audio_processor(audio_array, sr).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        print("Please check if the audio file exists and is in a supported format.")
        return

    print("\nInput:\n ", args.prompt, "\n\nOutputs:")
    for i in range(args.generations):
        # 可以調整生成參數，例如 temperature, top_k, top_p
        gen = model.generate(
            tokens, 
            audio_t, 
            max_new_tokens=args.max_new_tokens,
            greedy=True,  # 先使用greedy解碼測試
            top_k=10,     # 減小top_k
            top_p=0.8,    # 減小top_p
            temperature=0.3  # 降低temperature
        )
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        print(f"  >> Generation {i+1}: {out}")

if __name__ == "__main__":
    main()