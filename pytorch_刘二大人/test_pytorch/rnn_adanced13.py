import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gzip
import csv

class NameDateset(Dataset):#构建数据类型和对数据的操作
    def __init__(self,is_train_set):
        filename='./names_train.csv.gz' if is_train_set else './names_test.csv.gz'
        #打开训练数据集或者测试集，文件第一列为姓名，第二列为国家
        with gzip.open(filename,'rt') as f:#二进制方式打开gz文件，并且返回一个文件类型
            reader=csv.reader(f)#读取csv文件
            rows=list(reader)#将读取的内容换成list格式
            print(rows)
        self.names=[row[0] for row in rows]#名字列表
        self.len=len(self.names)#名字个数
        self.countrys=[row[1] for row in rows]#国家列表
        self.country_list=list(sorted(set(self.countrys)))#构建国家和类别序号之间的字典
        #将国家变成元组（无重复）并进行排序，获得排序后的所有国家类别列表
        self.country_dict=self.getCountryDict()#转换为词典
        self.country_num=len(self.country_list)#所有国家种类

    def __getitem__(self, item):#根据索引获取对应名字和国家类别标签（数字）
        return self.names[item],self.country_dict[self.countrys[item]]

    def __len__(self):#获得数据集的个数
        return self.len

    def getCountryDict(self):
        country_dict=dict()
        for idx,country_name in enumerate(self.country_list,0):
            country_dict[country_name]=idx
            #构建一个国家与数字对，这样可以通过预测出的数字找到国家类别
            #并且对应的数字实际上是和list的下标一样的
        return country_dict
    def idx2country(self,index):
        return self.country_list[index]
    def getCountriesNum(self):#获取所有国家类别数
        return self.country_num

HIDDEN_SIZE=100#隐藏层（以及输出）大小
BATCH_SIZE=256#每一批次训练的数据集大小
N_LAYER=2#GRU层数
N_EPOCHS=50#训练次数
N_CHARS=128 #用于构建嵌入层embedding，因为用ASCII码表示字符，所以一个onehot长度为128

trainSet=NameDateset(is_train_set=True)
trainLoader=DataLoader(trainSet,batch_size=BATCH_SIZE,shuffle=True)#加载数据
testSet=NameDateset(is_train_set=False)
testLoader=DataLoader(testSet,batch_size=BATCH_SIZE,shuffle=False)
#加载训练集和测试集

N_COUNTRY=trainSet.getCountriesNum()#国家类别数量，决定最终的输出维度

#构建模型
'''
输入维度
𝑖𝑛𝑝𝑢𝑡: (𝑠𝑒𝑞𝐿𝑒𝑛, 𝑏𝑎𝑡𝑐ℎ𝑆𝑖𝑧𝑒, ℎ𝑖𝑑𝑑𝑒𝑛𝑆𝑖𝑧𝑒)
hidden: (nLayers * nDirections, batchSize, hiddenSize)
输出维度
output: (seqLen, batchSize, hiddenSize * nDirections)
hidden: (nLayers * nDirections, batchSize, hiddenSize)
'''

class RNNClassifier(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layers=1,bidirectional=True):
        super(RNNClassifier,self).__init__()
        self.hidden_size=hidden_size
        self.n_layers=n_layers
        self.n_directions=2 if bidirectional else 1#方向数
        self.embedding=torch.nn.Embedding(input_size,hidden_size)#需要将原始的onehot（128维）转变为嵌入层，维度大小为隐藏层大小
        self.gru=torch.nn.GRU(hidden_size,hidden_size,n_layers,bidirectional=bidirectional)#输入输出可以看到都是hiddensize
        #因为嵌入层的处理，bidirectional代表是双向GRU还是单向GRU
        self.fc=torch.nn.Linear(hidden_size*self.n_directions,output_size)
        #构建一个线性层

    def _init_hidden(self,batch_size):
        hidden=torch.zeros(self.n_layers*self.n_directions,batch_size,self.hidden_size)
        #初始隐藏层为全零，第一个参数是层数*是否双向，代表着后面（batchsize，hiddensize）的多少
        return hidden

    def forward(self,input,seq_lengths):
        input=input.t()#转置，将输入从（batchsize，seqlenth）转变为（seqlength,batchsize），其中seqlength就是每个单词的被填充后的长度，现在是每一列代表一个单词了
        batch_size=input.size(1)#从0开始，第二个参数是batchsize
        hidden=self._init_hidden(batch_size)
        embedding=self.embedding(input)#运行后为（seqlength，batchsize，hiddesize），即将每一个下标用（ASCII的表示）转变为hiddensize大小的嵌入表示，
        # 原本每个下标需要用onehot表示（可以看博客加深了解）

        #pack them up
        gru_input=torch.nn.utils.rnn.pack_padded_sequence(embedding,seq_lengths)
        #把所有非零的（非填充的）数据都打包成一个二位矩阵，就是将一个batchsize的全打包
        # （相当于将这个维度扁平化了，但这必须要求按非零值个数由大到小在batchsize维度上从左到右排列），这使用make_Tensor实现
        #这个排列在送入embedding前就要进行，同时会专门记录下来每个单词的长度，以便知道每个时刻读取多少数据
        output,hidden=self.gru(gru_input,hidden)
        if self.n_directions==2:
            hidden_cat=torch.cat([hidden[-1],hidden[-2]],dim=1)#如果有两个方向，就将隐藏层拼接起来
        else:
            hidden_cat=hidden[-1]

        fc_output=self.fc(hidden_cat)#最后通过线性层变成想要的维度（分类数128）
        return fc_output

#将数据转化为Tensor，需要填充0以及按名字长度进行降序排序

def name2list(name):
    arr=[ord(c) for c in name]#将名字的每个字符换成对应ASCII值的列表
    return arr,len(arr)

def make_tensors(names,countries):
    sequences_and_lengths=[name2list(name) for name in names]#把所有名字进行变换
    name_sequences=[s1[0] for s1 in sequences_and_lengths]#去除所有的转换后的名字
    seq_lengths=torch.LongTensor([s1[1] for s1 in sequences_and_lengths])#取出所有名字的长度并转为为列表，然后转换成tensor
    countries=countries.long()
    # make tensor of name, BatchSize * seqLen
    # 他这里补零的方式先将所有的0 Tensor给初始化出来，然后在每行前面填充每个名字
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()#以最长单词长度为基准进行填充
    # print("seq_lengths.max:", seq_lengths.max())
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)#将前面部分填上名字表示

    # sort by length to use pack_padded_sequence
    # 将名字长度降序排列，并且返回降序之后的长度在原tensor中的小标perm_idx
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    # 这个Tensor中的类似于列表中切片的方法神奇啊，直接返回下标对应的元素，相等于排序了
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    # 返回排序之后名字Tensor，排序之后的名字长度Tensor，排序之后的国家名字Tensor
    return seq_tensor, seq_lengths, countries

classifier=RNNClassifier(N_CHARS,HIDDEN_SIZE,N_COUNTRY,N_LAYER)

criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(classifier.parameters(),lr=0.001)

import time
import math

def trainModel():
    def time_since(since):
        s=time.time()-since
        m=math.floor(s/60)
        s-=m*60
        return  '%d %ds' %(m,s)
    total_loss=0
    for i,(names,countries) in enumerate(trainLoader,1):
        inputs,seq_lengths,target=make_tensors(names,countries)#构建张量
        #output =classifier(inputs,seq_lengths)
        # 注意输出和目标的维度：Shape: torch.Size([256, 18]) torch.Size([256])
        output = classifier(inputs, seq_lengths)

        loss=criterion(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
        if i%10==0:
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(inputs)}/{len(trainSet)}] ', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
    return total_loss

def testModel():
    correct=0
    total=len(testSet)
    print("evaluating trained model ... ")
    with torch.no_grad():
        for i,(names,countries) in enumerate(testLoader):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths)
            # 注意这个keepdim的使用，为了直接和target计算loss
            pred = output.max(dim=1, keepdim=True)[1]
            # 注意这个view_as 和 eq
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')

    return correct / total

N_EPOCHS=50
start=time.time()
print("Training for %d epochs..." % N_EPOCHS)
acc_list = []
for epoch in range(1,N_EPOCHS+1):
    trainModel()
    acc=testModel()
    acc_list.append(acc)


import matplotlib.pyplot as plt
import numpy as np

epoch = np.arange(1, len(acc_list) + 1)
acc_list = np.array(acc_list)
plt.plot(epoch, acc_list)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.show()