import torch

class AudioQACollator(object):
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        audio_data = [item["audio"] for item in batch]
        genders = [item["gender"] for item in batch]
        transcriptions = [item["transcription"] for item in batch]
        major_emotions = [item["major_emotion"] for item in batch]

        # 堆叠音频
        audios = torch.stack(audio_data)

        # 创建输入序列
        input_sequences = []
        for i in range(len(transcriptions)):
            input_sequences.append(f"{transcriptions[i]}{genders[i]}{major_emotions[i]}")

        encoded_full_sequences = self.tokenizer.batch_encode_plus(
            input_sequences,
            padding="max_length",
            padding_side="left",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = encoded_full_sequences["input_ids"]
        attention_mask = encoded_full_sequences["attention_mask"]
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:].clone()
        labels[:, -1] = -100

        # 处理标签，预测所有内容（转录、性别、情绪）
        original_lengths = [len(self.tokenizer.encode(seq)) for seq in input_sequences]
        
        for i in range(len(batch)):
            if original_lengths[i] > self.max_length:
                labels[i, :] = -100
                continue
            
            # Don't mask anything - predict the entire sequence
            # The model will learn to generate transcription + gender + emotion
            # Labels are already set correctly from input_ids

        return {
            "audio": audios,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
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
            text = f"Transcription: {transcriptions[i]}. Gender: {genders[i]}. Emotion: {major_emotions[i]}."
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