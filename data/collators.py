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
        # 使用與 generate.py 中一致的提示格式
        self.prompt_text = "Question: What is said in this audio? Answer:"

    def __call__(self, batch):
        audio_data = [item["audio"] for item in batch]
        transcriptions = [item["transcription"] for item in batch]
        audios = torch.stack(audio_data)

        sequences_to_tokenize = []
        # 獲取提示文本的 token ID (不含特殊 token，因為它將作為前綴)
        # 注意：實際長度可能因 tokenizer 是否自動添加 BOS 等而異，這裡的 masking 策略需要小心
        prompt_tokens_for_length_estimation = self.tokenizer.encode(self.prompt_text, add_special_tokens=False)
        len_prompt_tokens = len(prompt_tokens_for_length_estimation)

        for trans in transcriptions:
            # 為轉錄文本添加 EOS token，教模型學會停止
            eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token else ""
            if eos_token and not trans.endswith(eos_token): # 檢查 eos_token 是否為空
                trans_with_eos = trans + eos_token
            else:
                trans_with_eos = trans
            
            # 構造 "提示 + 空格 + 答案" 的完整序列
            sequences_to_tokenize.append(self.prompt_text + " " + trans_with_eos)

        # 對完整序列進行分詞
        # 注意：`max_length` 應該是 `ALMConfig` 中為文本部分設定的總長度，例如 `alm_cfg.lm_max_length`
        encoded_sequences = self.tokenizer.batch_encode_plus(
            sequences_to_tokenize,
            padding="max_length", # 或 "longest"
            truncation=True,
            max_length=self.max_text_length, 
            return_tensors="pt",
            padding_side="left" # 與原始代碼保持一致的左填充
        )

        input_ids = encoded_sequences["input_ids"]
        attention_mask = encoded_sequences["attention_mask"]
        
        # 創建 labels，labels 與 input_ids 相同，但提示部分和填充部分會被 mask 掉 (設為 -100)
        labels = input_ids.clone()

        # Mask 掉提示部分 (prompt_text + " ") 在 labels 中的 token
        # 這一步比較複雜，因為左填充和 tokenizer 是否自動添加 BOS token 會影響提示的實際起始位置和長度
        # 一個簡化的策略是：假設提示部分在有效 token 中的長度是固定的。
        # 更穩健的方法是找到 "Answer:" 後面第一個 token 的位置。
        
        # 獲取 "提示 + 空格" 的 token，這部分在 loss 中應該被忽略
        prefix_to_mask_in_labels_text = self.prompt_text + " "
        # 分詞時不添加特殊token，以獲取純粹的內容token長度
        prefix_to_mask_tokens = self.tokenizer.encode(prefix_to_mask_in_labels_text, add_special_tokens=False)
        len_prefix_to_mask = len(prefix_to_mask_tokens)

        for i in range(input_ids.shape[0]):
            # 找到實際內容的起始位置 (跳過左填充的 pad_token 和 tokenizer 可能添加的 BOS token)
            actual_content_start_index = 0
            if self.tokenizer.pad_token_id is not None:
                non_pad_indices = (input_ids[i] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                if len(non_pad_indices) > 0:
                    actual_content_start_index = non_pad_indices[0].item()
            

            
            # 找到 "Answer: " 在 input_ids[i] 中的結束位置
            # 參考 https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/vocab.json 的 "ĠAnswer":19842,
            # "Ġ" 代表的是一個空格, "Answer" 的 token ID 是 21350, 實際上是根據 " Answer:" 進行分詞, 因此結果是 [19842, 42]
            # answer_marker_tokens = self.tokenizer.encode("Answer:", add_special_tokens=False) # 不包含空格, 結果是 [21350, 42]
            answer_marker_tokens = [19842, 42]
            marker_end_idx = -1
            for k in range(actual_content_start_index, input_ids.shape[1] - len(answer_marker_tokens) +1):
                if torch.equal(input_ids[i, k : k + len(answer_marker_tokens)], torch.tensor(answer_marker_tokens, device=input_ids.device)):
                    marker_end_idx = k + len(answer_marker_tokens) # "Answer:" 之後的空格也應 mask
                    # 如果 "Answer: " 後面有空格，則 marker_end_idx 需要再 +1 (如果空格是一個單獨的token)
                    # 檢查 "Answer: " 中的空格是否與下一個token合併
                    # 假設 "Answer: " (帶空格) 的長度是 len_prefix_to_mask
                    if marker_end_idx < labels.shape[1]:
                         labels[i, actual_content_start_index : marker_end_idx] = -100 # Mask prompt part
                    else: # 整個序列都是 prompt 或更短
                         labels[i, actual_content_start_index :] = -100
                    break
            if marker_end_idx == -1: # 如果找不到 "Answer:"，則 mask 整個有效序列 (可能有問題的樣本)
                labels[i, actual_content_start_index:] = -100


        # Mask 掉填充部分的 token
        if self.tokenizer.pad_token_id is not None:
            labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "audio": audios,
            "input_ids": input_ids,         # 包含 "提示 + 答案"
            "attention_mask": attention_mask, # 對應 "提示 + 答案"
            "labels": labels                # "提示"部分被 mask (值為 -100)，"答案"部分用於計算損失
        }

class SAVEECollator(object):  # https://huggingface.co/datasets/AbstractTTS/SAVEE
    def __init__(self, tokenizer, max_length=None): # 添加了 max_length 参数以便进行最大长度填充和截断
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch):
        audio_data = [item["audio"] for item in batch]
        genders = [item["gender"] for item in batch]
        transcriptions = [item["transcription"] for item in batch]
        major_emotions = [item["major_emotion"] for item in batch]

        # 堆叠音频数据
        audio_tensors = torch.stack(audio_data)

        # 创建包含转录、性别和情绪的目标文本序列
        # 模型将学习生成这种格式的文本
        # 例如："Transcription: [文本内容]. Gender: [性别]. Emotion: [情绪]."
        # 您可以根据模型的具体需求调整此格式
        full_target_texts = []
        for i in range(len(batch)):
            # text = f"Transcription: {transcriptions[i]}. Gender: {genders[i]}. Emotion: {major_emotions[i]}."
            text = f"Transcription: {transcriptions[i]}. Gender: {genders[i]}."
            full_target_texts.append(text)
        
        padding_strategy = "longest"
        tokenizer_args = {
            "padding": padding_strategy,
            "return_tensors": "pt",
            "padding_side": "left", # 与文件中的其他 collator 保持一致
        }

        if self.max_length:
            tokenizer_args["max_length"] = self.max_length
            tokenizer_args["truncation"] = True
            # 如果指定了 max_length，确保使用 "max_length" 策略进行填充
            if padding_strategy == "longest":
                 tokenizer_args["padding"] = "max_length"

        # 对合并后的文本序列进行分词
        encoded_sequences = self.tokenizer.batch_encode_plus(
            full_target_texts,
            **tokenizer_args
        )
        
        input_ids = encoded_sequences['input_ids']
        attention_mask = encoded_sequences['attention_mask']
        
        # 在自回归语言模型中，标签通常就是 input_ids 本身
        # 填充部分的 token 在计算损失时应被忽略，因此将其设置为 -100
        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        # 如果分词器没有 pad_token_id，但发生了填充（例如，填充到 max_length 时使用默认ID如0），
        # 这种情况需要根据分词器的具体行为小心处理。
        # 这里假设如果启用了填充，分词器中已正确设置了 pad_token_id。

        return {
            "audio": audio_tensors,                   # 音频张量 (修正了原代码中的 "audios" 并使用了正确的变量名)
            "input_ids": input_ids,                   # 分词后的目标文本序列 (包含转录、性别、情绪)
            "attention_mask": attention_mask,         # 对应 input_ids 的注意力掩码
            "labels": labels,                         # 用于训练语言模型部分的标签 (填充部分已设为 -100)
        }