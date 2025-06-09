class AudioQACollator(object):
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        audio_data = [item["audio"] for item in batch]
        genders = [item["gender"] for item in batch]
        transcriptions = [item["transcription"] for item in batch]
        # major_emotions = [item["major_emotion"] for item in batch]

        # 堆叠音频
        audios = torch.stack(audio_data)

        # 创建输入序列
        input_sequences = []
        for i in range(len(transcriptions)):
            # input_sequences.append(f"{transcriptions[i]}{genders[i]}{major_emotions[i]}")
            input_sequences.append(f"{transcriptions[i]}{genders[i]}")

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