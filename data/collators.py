import torch

# 這是一個專為對齊預訓練設計的、更簡單的 Collator
class AlignmentCollator(object):
    def __init__(self, tokenizer, text_max_length, audio_processor):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.audio_processor = audio_processor # 假設您有音頻處理器
        # 完整的句子模板
        self.prompt_template = "Question: What is said in this audio? Answer: {}"

    def __call__(self, batch):
        # 1. 處理音頻
        # audio_features = self.audio_processor([item["audio"]["array"] for item in batch], return_tensors="pt")
        # 這裡簡化為您已有的 Torch 張量
        audios = torch.stack([item["audio"] for item in batch])

        # 2. 處理文本 (將轉錄稿填入模板)
        full_texts = [self.prompt_template.format(item["transcription"]) for item in batch]
        
        # 3. 分詞，注意這裡不需要複雜的 label masking
        encoded_texts = self.tokenizer.batch_encode_plus(
            full_texts,
            padding="max_length", # 或者 "longest"
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt"
        )
        
        # 返回對齊任務需要的數據
        return {
            "audio": audios,
            "input_ids": encoded_texts["input_ids"],
            "attention_mask": encoded_texts["attention_mask"]
        }


class AudioQACollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.chat_template is None:
            raise ValueError("The tokenizer must have a chat_template.")
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids('<AUDIO>')

    def __call__(self, batch):
        audio_data = [item["audio"] for item in batch]
        transcriptions = [item["transcription"] for item in batch]
        audios = torch.stack(audio_data)

        all_final_input_ids = []
        all_final_labels = []

        for trans in transcriptions:
            eos_token = self.tokenizer.eos_token or ""
            if not trans.strip().endswith(eos_token):
                trans += eos_token
            
            messages = [
                {"role": "user", "content": "What is said in this audio? <AUDIO>"},
                {"role": "assistant", "content": trans}
            ]

            # 1. Tokenize 完整的對話，得到 input_ids
            full_input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )

            # 2. 找到答案的起始位置
            assistant_messages = messages[1:2]
            assistant_ids = self.tokenizer.apply_chat_template(
                assistant_messages, tokenize=True, add_generation_prompt=True
            )

            all_final_input_ids.append(torch.tensor(full_input_ids))
            all_final_labels.append(torch.tensor(assistant_ids))

        # 5. 填充整個批次，使其長度一致
        input_ids = torch.nn.utils.rnn.pad_sequence(
            all_final_input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            all_final_labels, 
            batch_first=True, 
            padding_value=-100
        )
        
        # 6. 創建 attention_mask
        # 注意：attention_mask 是基於填充後的 input_ids 創建的
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "audio": audios,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
