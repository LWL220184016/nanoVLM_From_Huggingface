import sys
from pathlib import Path
# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.processors import get_tokenizer
from models.config import ALMConfig


def main():
    cfg = ALMConfig()

    tokenizer = get_tokenizer(cfg.lm_tokenizer)

    eos_token = tokenizer.eos_token or ""


    messages = [
        {"role": "user", "content": "What is said in this audio?"},
        {"role": "assistant", "content": f"Nothing {eos_token}"}
    ]

    # 2. 使用 apply_chat_template 生成完整的 input_ids
    # 這會處理好所有特殊 token 和格式
    chat_template = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False # 我們提供了完整的對話，所以設為 False
    )
    print("Chat template:\n", chat_template)

    encoded = tokenizer.batch_encode_plus([chat_template], return_tensors="pt")
    input_ids = encoded["input_ids"]

    print("Input IDs:", input_ids)


if __name__ == "__main__":
    main()

# Output:
# Chat template:
#  <|im_start|>user
# What is said in this audio?<|im_end|>
# <|im_start|>assistant
# Nothing <|endoftext|><|im_end|>

# Input IDs: tensor([[    1,  4093,   198,  1780,   314,  1137,   281,   451,  8389,    47,
#              2,   198,     1,   520,  9531,   198, 32842,   216,     0,     2,
#            198]])