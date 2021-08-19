# -*- coding:utf-8 -*-

import collections
import json

"""
index=0
vocab = collections.OrderedDict()
vocab_file="/home/hoya/notebooks/syoung21.yoo/pytorch-bert-crf-ner/bert1_aihub/original/vocab.korean_morp_o.list"
with open(vocab_file, "r", encoding="utf-8") as reader:
    while True:
        token = reader.readline()
        if not token:
            break
				
        if token.find('n_iters=') == 0 or token.find('max_length=') == 0 :
            continue
        token = token.split('\t')[0]
        token=token.split("/")[0]
	  
        token = token.strip()
        vocab[token] = index
        index += 1
reader.close()

with open("./aihub_vocab_dict.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)

f.close()
"""

with open("/home/hoya/notebooks/syoung21.yoo/pytorch-bert-crf-ner/bert1_aihub/jvocab_aihub1.json", "r", encoding="utf-8") as f:
    json_data=json.load(f)
    dict={}
    dict=json_data["token_to_idx"]
    print(type(json_data["token_to_idx"]))

with open("./aihub_vocab_dict.json", "w", encoding="utf-8") as result:
    json.dump(dict, result, ensure_ascii=False)

f.close()
result.close()