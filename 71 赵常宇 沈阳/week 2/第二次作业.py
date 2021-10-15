import torch
import torch.nn as nn
import random
import numpy as np
import json
import matplotlib.pyplot as plt #绘图
'''--------数据的生成，包括测试、训练、以及标签数据--------'''
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz123"  #字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index + 1   #每个字对应一个序号
    vocab['unk'] = len(vocab)+1
    return vocab
#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):      #vocab 6
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #A类样本
    if set("abc") & set(x) and not set("123") & set(x):
        y = 0
    #B类样本
    elif not set("abc") & set(x) and set("123") & set(x):
        y = 1
    #C类样本
    else:
        y = 2
    x = [vocab.get(word, vocab['unk']) for word in x]        #将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    """
        输入需要的样本数量，建立数据集
        :param sample_length: 样本个数
        :param sentence_length: 单个样本的长度，比如6个字符
        :return: tensor格式的数据、tensor格式的标签
        """
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y) #返回longtensor类型的
#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model
"""*******************      神经网络的模型搭建      *******************"""
class TorchModel(nn.Module):
    def __init__(self, input_dim, sentence_length, vocab): #
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim)
        self.layer = nn.Linear(input_dim, input_dim)
        self.pool = nn.MaxPool1d(sentence_length)
        self.classify = nn.Linear(input_dim, 3) #这是一个多分类的任务 0 1 2
        self.activation = torch.sigmoid     #sigmoid做激活函数
        self.dropout = nn.Dropout(0.4)
        self.loss = nn.functional.cross_entropy  #loss采用均方差损失
    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  #input shape:(batch_size, sen_len) (10,6)
        x = self.layer(x)      #input shape:(batch_size, sen_len, input_dim) (10,6,20)
        x = self.dropout(x)    #input shape:(batch_size, sen_len, input_dim)
        x = self.activation(x) #input shape:(batch_size, sen_len, input_dim)
        x = self.pool(x.transpose(1,2)).squeeze() #input shape:(batch_size, sen_len, input_dim)
        x = self.classify(x)   #input shape:(batch_size, input_dim)  (10,6)
        y_pred = self.activation(x)               #input shape:(batch_size, 1) (10,1)
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred
def build_model(vocab, char_dim, sentence_length):#vocab ，20， 6
    model = TorchModel(char_dim, sentence_length, vocab)
    return model
#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    total = 200 #测试样本数量
    x, y = build_dataset(total, vocab, sample_length)   #建立200个用于测试的样本
    y = y.squeeze()
    print("A类样本数量：%d, B类样本数量：%d, C类样本数量：%d"%(y.tolist().count(0), y.tolist().count(1), y.tolist().count(2)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1   #正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d / %d, 正确率：%f"%(correct, total, correct/(correct+wrong)))
    return correct/(correct+wrong)
"""*******************          【三、最终预测】        *******************"""
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
        print(int(torch.argmax(result[i])), input_string, result[i]) #打印结果
#设置main函数
def main():
    epoch_num = 15        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 1000   #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    vocab = build_vocab()       #建立字表
    model = build_model(vocab, char_dim, sentence_length)    #建立模型
    optim = torch.optim.Adam(model.parameters(), lr=0.005)   #建立优化器
    log = []
    for epoch in range(epoch_num):
        model.train()  #训练过程中会在程序上方添加一句model.train()，作用是启用batch normalization和drop out。
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
    plt.plot(range(len(log)), [l[0] for l in log])  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log])  #画loss曲线
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return
if __name__ == "__main__":
    #main()
    test_strings = ["123def", "123ghj", "rbweqg", "nlhdww"]
    predict("model.pth", "vocab.json", test_strings)

