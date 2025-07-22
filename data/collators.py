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
    def __init__(self, tokenizer, max_text_length): # max_text_length 是 "提示+答案" 的總長度
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        # 移除 self.prompt_text，改用聊天模板
        # 確保 tokenizer 有 chat_template
        if self.tokenizer.chat_template is None:
            raise ValueError(
                "The tokenizer must have a chat_template. "
                "You can either manually set it or use a model that has one."
            )

    def __call__(self, batch):
        audio_data = [item["audio"] for item in batch]
        transcriptions = [item["transcription"] for item in batch]
        audios = torch.stack(audio_data)

        all_input_ids = []
        all_labels = []

        for trans in transcriptions:
            # 1. 構造對話消息
            # 確保答案以 EOS token 結尾，引導模型學會停止生成
            eos_token = self.tokenizer.eos_token or ""
            if not trans.strip().endswith(eos_token):
                trans += eos_token
            
            messages = [
                {"role": "user", "content": "What is said in this audio?"},
                {"role": "assistant", "content": trans}
            ]

            # 2. 使用 apply_chat_template 生成完整的 input_ids
            # 這會處理好所有特殊 token 和格式
            full_input_ids = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=False # 我們提供了完整的對話，所以設為 False
            )

            # 3. 創建 labels，只保留 assistant 回答的部分
            # 為了找到答案的起始點，我們只模板化用戶的部分
            # add_generation_prompt=True 會在用戶消息後添加提示，讓模型準備好生成回答
            # 例如，它可能會添加 `[/INST]` 或 `<|start_header_id|>assistant<|end_header_id|>\n\n`
            prompt_only_ids = self.tokenizer.apply_chat_template(
                messages[:-1], # 只包含 user 消息
                tokenize=True,
                add_generation_prompt=True
            )
            
            labels = [-100] * len(full_input_ids)
            
            # assistant 的回答部分就是 full_input_ids 中 prompt_only_ids 後面的部分
            answer_start_index = len(prompt_only_ids)
            labels[answer_start_index:] = full_input_ids[answer_start_index:]
            
            all_input_ids.append(torch.tensor(full_input_ids))
            all_labels.append(torch.tensor(labels))

        # 4. 對 batch 進行填充
        # 因為我們是逐個處理的，所以需要手動填充
        # 注意：Hugging Face 的 DataCollatorForSeq2Seq 或類似工具可以自動化這一步
        input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            all_labels, 
            batch_first=True, 
            padding_value=-100 # label 的填充值是 -100
        )

        # 5. 截斷到最大長度
        input_ids = input_ids[:, :self.max_text_length]
        labels = labels[:, :self.max_text_length]
        
        # 6. 創建 attention_mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "audio": audios,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    