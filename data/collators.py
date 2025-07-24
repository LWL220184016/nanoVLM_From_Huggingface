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
    def __init__(self, tokenizer, max_text_length, audio_patches):
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.audio_patches = audio_patches # 音訊嵌入的長度
        if self.tokenizer.chat_template is None:
            raise ValueError("The tokenizer must have a chat_template.")

    def __call__(self, batch):
        audio_data = [item["audio"] for item in batch]
        transcriptions = [item["transcription"] for item in batch]
        audios = torch.stack(audio_data)

        all_input_ids = []
        all_labels = []

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

            # 2. 找到 <AUDIO> token 和答案的起始位置
            # 我們需要一個沒有答案的版本來定位
            prompt_messages = messages[:-1]
            prompt_ids = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=True, add_generation_prompt=True
            )
            
            try:
                # 找到 <AUDIO> token 在 input_ids 中的索引
                audio_token_idx = full_input_ids.index(self.tokenizer.convert_tokens_to_ids('<AUDIO>'))
            except ValueError:
                # 如果模板不包含 <AUDIO>，這是一個錯誤
                raise ValueError("'<AUDIO>' token not found in the tokenized prompt. Check your chat template.")

            # 答案開始的索引
            answer_start_index = len(prompt_ids)

            # 3. 創建與最終 logits 維度匹配的 labels
            # 初始標籤，全部忽略
            labels = [-100] * len(full_input_ids)
            
            # 將答案部分的標籤替換為真實的 token ID
            labels[answer_start_index:] = full_input_ids[answer_start_index:]

            # 4. *** 核心修改：為音訊嵌入調整 labels ***
            # 在 <AUDIO> token 的位置，插入 N 個 -100，N = 音訊 patch 數量
            # 這模擬了 forward 函數中的嵌入替換過程
            audio_padding = [-100] * self.audio_patches
            # 將 <AUDIO> token 對應的單個 -100 標籤替換為 N 個 -100
            final_labels = labels[:audio_token_idx] + audio_padding + labels[audio_token_idx + 1:]
            
            all_input_ids.append(torch.tensor(full_input_ids))
            all_labels.append(torch.tensor(final_labels))

        # 5. 填充 batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            all_labels, 
            batch_first=True, 
            padding_value=-100
        )

        # 6. 截斷到最大長度
        # 注意：這裡的最大長度需要考慮到音訊 patch 的增加
        final_max_length = self.max_text_length + self.audio_patches
        input_ids = input_ids[:, :self.max_text_length] # input_ids 保持原長度
        labels = labels[:, :final_max_length]          # labels 擴展到新長度
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "audio": audios,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
