from data.processors import get_tokenizer
from datasets import load_dataset
from models.config import ALMConfig, TrainConfig

def debug_tokenizer_dataset_compatibility():
    """檢查tokenizer和數據集的兼容性"""
    lm_config = ALMConfig()
    train_config = TrainConfig()
    # 載入tokenizer
    tokenizer = get_tokenizer(lm_config.lm_tokenizer, lm_config.lm_vocab_size)

    print("--------------------------------------------------------")
    print("Tokenizer loaded successfully.")
    print(f"Tokenizer type: {type(tokenizer)}")
    print(f"Tokenizer name: {tokenizer.name_or_path}")
    print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
    print(f"Tokenizer padding side: {tokenizer.padding_side}")
    print(f"Tokenizer truncation side: {tokenizer.truncation_side}")
    print(f"Tokenizer eos token: {tokenizer.eos_token}")
    print(f"Tokenizer bos token: {tokenizer.bos_token}")
    print(f"Tokenizer unk token: {tokenizer.unk_token}")
    print(f"Tokenizer pad token: {tokenizer.pad_token}")
    print(f"Tokenizer additional special tokens: {tokenizer.additional_special_tokens}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Tokenizer model max length: {tokenizer.model_max_length}")
    
    # 載入數據集樣本
    dataset = load_dataset(
        path = train_config.train_dataset_path, 
        name = train_config.train_dataset_name
    )
    print("\n--------------------------------------------------------")
    print("Dataset loaded successfully.")
    print(f"Dataset loaded: {train_config.train_dataset_path}/{train_config.train_dataset_name}")
    print(f"Dataset type: {type(dataset)}")
    print(f"Dataset features: {dataset['train'].features}")
    print(f"Dataset keys: {dataset.keys()}")
    
    # 檢查幾個樣本
    for i in range(min(5, len(dataset['train']))):
        sample = dataset['train'][i]
        print(f"\nSample {i}:")
        print(f"Keys: {sample.keys()}")
        
        # 如果有文本字段，檢查tokenization
        for key in sample.keys():
            if isinstance(sample[key], str) and len(sample[key]) > 0:
                print(f"Text field '{key}': {sample[key][:100]}...")
                
                # Tokenize並檢查
                tokens = tokenizer.encode(sample[key])
                print(f"Tokenized length: {len(tokens)}")
                print(f"Token range: {min(tokens)} to {max(tokens)}")
                
                # 檢查是否有超出詞彙表的token
                invalid_tokens = [t for t in tokens if t >= tokenizer.vocab_size]
                if invalid_tokens:
                    print(f"WARNING: Invalid tokens found: {invalid_tokens}")
                
                # 解碼回來檢查
                decoded = tokenizer.decode(tokens)
                print(f"Decoded matches original: {decoded == sample[key]}")

if __name__ == "__main__":
    debug_tokenizer_dataset_compatibility()