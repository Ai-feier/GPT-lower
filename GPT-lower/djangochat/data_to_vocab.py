import json

'''
    处理数据集，获得vocab文件
'''

with open('./dataset/LCCC-base_test.json', 'rb') as f:
    data1 = json.load(f)
with open('./dataset/LCCC-base_train.json', 'rb') as f:
    data2 = json.load(f)
with open('./dataset/LCCC-base_valid.json', 'rb') as f:
    data3 = json.load(f)

with open('./model/bert_chinese/vocab.txt', 'r', encoding='utf-8') as f:
    data4 = f.read()

vocab_list = []
vocab_list.extend(list(data4.split('\n')))
for idx, str in enumerate(data1):
    for s in str:
        if len(str) <= 3:
            vocab_list.extend(list(s))

for idx, str in enumerate(data2):
    for s in str:
        if len(str) <= 3:
            vocab_list.extend(list(s))

for idx, str in enumerate(data3):
    for s in str:
        if len(str) <= 3:
            vocab_list.extend(list(s))


vocab_list = sorted(list(set(vocab_list)))

vocab_size = 29994
# vocab_list = vocab_list[700:vocab_size+700]
vocab_list.insert(0, '[speaker2]')
vocab_list.insert(0, '[speaker1]')
vocab_list.insert(0, '[UNK]')
vocab_list.insert(0, '[SEP]')
vocab_list.insert(0, '[PAD]')
vocab_list.insert(0, '[CLS]')
with open('./model/bert/vocab.txt', 'w', encoding='utf-8') as f:
    for token in vocab_list:
        f.write(f'{token}\n')

