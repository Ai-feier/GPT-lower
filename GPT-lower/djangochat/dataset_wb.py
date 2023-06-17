import json

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

# 加载json二维列表对话集
class LCCCdiskDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_history=15, batch_first=True, lm_labels=True, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.data = self.load_dataset()
        self.max_history = max_history
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels

    def __getitem__(self, index):
        data = self.data[index]
        tokenizer = self.tokenizer
        sentencepiece = ''
        '''
            获取每个json列表中的语料
            并为语料添加说话人
            并添加顺序与分词
        '''
        for i, sentence in enumerate(data):
            if i % 2 == 0:
                sentencepiece += "[speaker1]"
                sentencepiece += sentence
            else:
                sentencepiece += "[speaker2]"
                sentencepiece += sentence
            if i < len(data) - 1:
                sentencepiece += '[SEP]'
        '''
            将处理好的文本文件转化为模型可识别的输入
        '''
        return self.process(sentencepiece)

    def __len__(self):
        return len(self.data)


    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["attention_mask"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, labels

    def process(self, sentence, with_eos=True):

        return self.tokenizer(sentence)


    def load_dataset(self):
        with open(self.data_path, 'rb') as f:
            data = json.load(f)
        return data
