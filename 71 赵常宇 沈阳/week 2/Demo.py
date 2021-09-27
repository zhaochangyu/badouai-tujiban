#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json


#模型判断文本中是否有特定字符出现
#如果有abc的任何一个出现规定正样本否则规定为负样本
#修改后xyz

class TorchModel(nn.Module):     #继承父类nn.Module
    def __init__(self, input_dim, sentence_length, vocab):
        super(TorchModel, self).__init__() #初始化
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim)   #共有27对
        self.layer = nn.Linear(input_dim, input_dim)
        self.pool = nn.AvgPool1d(sentence_length)
        self.classify = nn.Linear(input_dim, 1)
        self.activation = torch.sigmoid      #sigmoid做激活函数
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.mse_loss   #loss采用均方差损失


    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  #input shape:(batch_size, sen_len)
        x = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)
        x = self.dropout(x)    #input shape:(batch_size, sen_len, input_dim)
        x = self.activation(x) #input shape:(batch_size, sen_len, input_dim)
        x = self.pool(x.transpose(1,2)).squeeze() #input shape:(batch_size, sen_len, input_dim)
        x = self.classify(x)   #input shape:(batch_size, input_dim)
        y_pred = self.activation(x)               #input shape:(batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

#字符集随便挑了一些汉字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index + 1   # 每个字对应一个序号 a对应1  b对应2 以此类推 unk对应27
    vocab['unk'] = len(vocab)+1
    return vocab
'''构建的字典如下{
  "a": 1,"b": 2, "c": 3,"d": 4,"e": 5,"f": 6,"g": 7,"h": 8,"i": 9,"j": 10, "k": 11, "l": 12,"m": 13,
  "n": 14,"o": 15,"p": 16,"q": 17,"r": 18, "s": 19,"t": 20,"u": 21,"v": 22,"w": 23,"x": 24,"y": 25,"z": 26,"unk": 27}
  vocab
'''

# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 反之为负样本
def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 随机从字表选取sentence_length个字，可能重复
    #if set("abc") & set(x):
    if set("xyz") & set(x):
        #  指定哪些字出现时为正样本
        y = 1
    else:
        # 指定字都未出现，则为负样本
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]
    # 将字转换成序号，为了做embedding
    return x, y
# get()函数的作用
# 返回字典中指定键的值(vocab是一个字典)
# 语法：dict.get(key, default=None)
# key--字典中要查找的键
# default--如果指定键的值不存在，则返回该默认值
# set()函数作用
# x = set('runoob')
# >>> y = set('google')
# >>> x, y
# (set(['b', 'r', 'u', 'o', 'n']), set(['e', 'o', 'g', 'l']))
# >>> x & y         # 交集
# set(['o'])
# >>> x | y         # 并集
# set(['b', 'e', 'g', 'l', 'o', 'n', 'r', 'u'])
# >>> x - y         # 差集
# set(['r', 'b', 'u', 'n'])


#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    sunnum=200       #建立200个用于测试的样本
    x, y = build_dataset(sunnum, vocab, sample_length)
    print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), sunnum - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1   #负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1   #正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    epoch_num = 10        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 1000   #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    vocab = build_vocab()       #建立字表
    model = build_model(vocab, char_dim, sentence_length)    #建立模型
    optim = torch.optim.Adam(model.parameters(), lr=0.005)   #建立优化器
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构建一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#最终预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)    #建立模型
    model.load_state_dict(torch.load(model_path))       #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式，不使用dropout
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print(round(float(result[i])), input_string, result[i]) #打印结果


if __name__ == "__main__":
    main()
    test_strings = ["asdafg", "cadffg", "rqwwxy", "efgtrh"]
    predict("model.pth", "vocab.json", test_strings)