
import  torch

input_size=4#对应x的onehot表示大小（词汇表大小）
hidden_size=4
#最后输出的类型共4个字母，rnn每一步都会输出一个四维的向量（行数会随着向后运算而增加），并通过softmax计算出可能的各字母概率
#最后一个隐藏状态会得到最终的对应的seqlen（或者rnn选择的大小）X4的矩阵，每个位置都是概率的大熊啊
batch_size=1 #每一输入的样本的数量

idx2char=['e','h','l','o']

x_data=[1,0,2,2,3]#就是hello
y_data=[3,1,2,3,2]#输出是判断的结果，就是属于e，h,l,o哪个类型
one_hot_lookup=[
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]]#one-hot会英于词汇表idx2char
x_one_hot=[one_hot_lookup[x] for x in x_data]
#生成了hello的one-hot表达形式，每一行代表一个字母的one-hot

inputs=torch.Tensor(x_one_hot).view(-1,batch_size,input_size)
#inputs维度为（seqlen,batchsize,inputsize）
#自动算出-1代表的序列长度这里应为5，对应hello的长度
#同时，对于rnn每一步输入一个字母，所以这样的顺序设置实际上可以便于获取每一步的输入

#对于本测试，batchsize就是一个字母，所以为1，
# 字母用4维向量表示，所以input_szie维4
#输出的类别判断也只有四种，所以hidden——size为4
#当然，本代码只是展示，并不是真正的一个任务

labels=torch.LongTensor(y_data).view(-1,1)#变成列是为了每个样本对应起来方便，
#而若是用torch.nn.RNN则不需要进行变换，因为是一次性输入的
print(labels)
#将结果改成4行一列，LongTensor是默认64位整数
#输出维度（hidden的不是label）为，（batchsize,inputsize

class Model(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size):
        super(Model,self).__init__()
        self.input_size=input_size
        self.batch_size=batch_size
        self.hidden_size=hidden_size
        self.rnncell=torch.nn.RNNCell(input_size=self.input_size,
                                      hidden_size=self.hidden_size)#rnn的神经网络层
        #使用rnncell需要说明输入大小（x大小）和输出大小（隐藏层大小）
        #若使用torch.nn.RNN，还需要一个num_layers的参数，用来确定是几层RNN（指上下）
    def forward(self,input,hidden):
        hidden=self.rnncell(input,hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size,self.hidden_size)
    # hidden0，随机初始化或者初始化为0向量,
    # 一定要注意，hidden和x的维度都要有batchsize,
    # batchsize只有在构建h0的时候才有用，如果已经提前准备了可以山区batchsize，x的batchszize也已经在准备数据时算出（见inputs）
    # 即一次传入的样本的数量
model=Model(input_size,hidden_size,batch_size)

criterion=torch.nn.CrossEntropyLoss()#交叉熵损失函数，对于分类很有用
optimizer=torch.optim.Adam(model.parameters(),lr=0.1)

for epoch in range(15):
    loss=0

    hidden=model.init_hidden()
    print('预测字符串：',end='')#不要换行
    for input,label in zip(inputs,labels):
        hidden=model(input,hidden)#同样会在call中自动调用forward所以不用.forward()
        loss+=criterion(hidden,label)
        #这个loss是求出每一项输入的损失之和，然后才是一次的损失
        #这里是构建了计算图，所以不要用loss.item()
        #这里lable存的时对应字符下标，hidden则是四种的概率，
        # 而交叉熵损失函数会乘以onehot获得概率最高的的下标进行计算损失。
        _,idx=hidden.max(dim=1)#求按行出每一个样本的最大概率类别，第一个是概率，第二个是下标
        print(idx2char[idx.item()],end='')#这次输出中不要换行

    optimizer.zero_grad()
    loss.backward()
    #每次训练周期输出计算一次梯度，前面哪个for循环只不过是在不断的输入数据
    #所以rnn的循环次数应该是看输入数据的长度
    optimizer.step()
    if (epoch+1)//15==0:
        print(loss.item())
'''
如果用RNN需要在构建模型时多一个ums_layers，并且hidde的维度也要加上改项，forward输出
也变成了（seqlen*batchsize,hiddensize ）以便化为两维矩阵
详细的看https://blog.csdn.net/Miracle_ps/article/details/114493682

训练时也不需要一个个输入了，直接输入整个inputs即可。
还有个embedding的方法不写了，直接看一下改变了啥就行了，最后一节在详细解释
需要说明的是最后通过一个线性层是为了输出时，隐藏层（指的是列的维度，每一行是代表分类的onehot，行数代表输出结果个数）与分类的数量一致（因为线性层只会改变最后一个维度，这样才可以转换成对应的字符，见网课），nn.Linear(hidden_size, num_class)
'''