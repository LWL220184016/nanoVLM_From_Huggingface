import torch
from PIL import Image

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_audio_processor

from torch.utils import benchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def generate_tokens(tokens, audio):
    gen = model.generate(tokens, audio, max_new_tokens=100)

if __name__ == "__main__":
    model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-222M").to(device)
    model.eval()
    
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    audio_processor = get_audio_processor(model.cfg.audio_sample_rate_size)

    text = "What is this?"
    template = f"Question: {text} Answer:"
    encoded_batch = tokenizer.batch_encode_plus([template], return_tensors="pt")
    tokens = encoded_batch['input_ids'].to(device)

    audio_path = 'assets/audio.png'
    audio = Image.open(audio_path)
    audio = audio_processor(audio)
    audio = audio.unsqueeze(0).to(device)

    time = benchmark.Timer(
        stmt="generate_tokens(tokens, audio)",
        setup='from __main__ import generate_tokens',
        globals={"tokens": tokens, "audio": audio},
        num_threads=torch.get_num_threads(),
    )

    print(time.timeit(10))
