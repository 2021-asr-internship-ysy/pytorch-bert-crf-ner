from tokenization_morp import*
import collections
import json

BTK=BertTokenizer("/home/hoya/notebooks/syoung21.yoo/pytorch-bert-crf-ner/bert1_aihub/original/vocab.korean_morp_o.list")

BasicTK=BTK.basic_tokenizer
wpTK=BTK.wordpiece_tokenizer 
#print(BertTK.vocab)

"""
with open("./vocab_test.json", 'w', encoding="utf-8") as vocab_file:
    json.dump(wpTK.vocab, vocab_file, ensure_ascii=False)
    if "개발" in wpTK.vocab:
        print(true)

"""
a=0
for key, val in wpTK.vocab.items():
    a=a+1
    if(a>500):  break
    if(key=="개발"):
        print(key, val)
        break
#vocab_file.close()


test_txt=open("/home/hoya/notebooks/syoung21.yoo/pytorch-bert-crf-ner/bert1_aihub/test/test_txt.txt", "r", encoding='utf-8')
line = test_txt.readline()

result1=BasicTK.tokenize(line)

morp_result=[]
for word in result1:
    w_result=wpTK.tokenize(word)
    for f in w_result:
        morp_result.append(f)

result2=BTK.tokenize(line)

print(result1)
print(result2)
print(morp_result)

test_txt.close()
