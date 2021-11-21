import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from  torch.utils.data import  DataLoader
from  sklearn.metrics import  mean_squared_error as MSE
raw_data = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)#.gz会自动系解析到里面csv文件
X = raw_data[:, :-1]
Y = raw_data[:, [-1]]
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3)#划分数据集
'''
       这里创建数据类，是因为当数据很多时，就不可能把所有数据全部导入内存中操作，
       那么就可以对类中的获取元素 和长度的函数进行设计，当需要时才读取对应部分，这样可以更合理的利用内存
       当然，下面的例子中还是都读入内存中
'''
class DiaDateset(Dataset):#用来创建一个数据类
    def __init__(self,data,label):
        #xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)  # 可以用来读取txt和csv
        self.len=data.shape[0]#求出行数，即样本数
        self.x_data = torch.from_numpy(data)  # -1代表最后一列，所以由于左闭右开，实际上到了倒数第二列
        self.y_data = torch.from_numpy(label)  # 若提示没有包是因为缺少pylint依赖

    def __getitem__(self, item):
        return self.x_data[item],self.y_data[item]#获取对应训练样本的数据和标签

    def __len__(self):
        return self.len #因为数据比较小，所以都读入内存了，求长度比较方便

train_dataset= DiaDateset(xtrain,ytrain)#创建数据类对象
#实际上有些数据集是封装好的可以直接用，比如dataset.MNIST(参数)可以直接加载数据集
train_loader =DataLoader( #加载数据
    dataset=train_dataset,
    batch_size=32,#一个minibatch的大小
    shuffle=True, #是否打乱
    num_workers=2#读取数据的进程数
)#会自动调用dataset中的__getitem__返回数据
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1=torch.nn.Linear(8,6)
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,1)
        self.sigmoid=torch.nn.Sigmoid()#这里的sigmoid是一层网络
    def forward(self,x):
        x=self.sigmoid(self.linear1(x))
        x=self.sigmoid(self.linear2(x))
        x=self.sigmoid(self.linear3(x))
        return x

model=Model()
criterion=torch.nn.BCELoss(reduction='mean')
#这里相比于前面求了平均值，所以除了N，所以求导后梯度较小，需要设置大一点的学习率
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)

def train(epoch):
    train_loss=0.0
    count=0
    #每一次i执行一次minibatch大小的前馈，当所有i执行完就是执行完了一次epoch训练（所有训练姐训练了一遍），相比于每一次都是一个样本的进行训练更好
    for i, data in enumerate(train_loader,0):#进行迭代，从i=0开始执行，将读取的对应i的数据存入data中,
        inputs,lables=data#dataloader会自动将数据集转化为张量
        y_pred=model(inputs)
        loss=criterion(y_pred,lables)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()#求和每一步的损失
        count=i
    if epoch%2000 ==1999:#epoch从0开始，所以正好是两千次输出一次平均损失
        print(train_loss/count,end=',')#用来设置以什么结尾，即换行

def test():
    with torch.no_grad():#测试不需要进行梯度下降
        y_pred=model(xtest)
        mse=MSE(ytest,y_pred)
        print(mse)
        #y_pred_label=torch.where(y_pred>0.5,torch.tensor([1.0]),torch.tensor([0.0]))
'''
if __name__ =='__main__':#由于windows和linux线程的差异，需要加这一行才可以正确执行，linux应该不用
    for epoch in range(1000):
        for i, data in enumerate(train_loader,0):#进行迭代，从i=0开始执行，将读取的数据存入data中
            inputs,lables=data#dataloader会自动将数据集转化为张量
            y_pred=model(inputs)
            loss=criterion(y_pred,lables)
            print(epoch,i,loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

'''
if __name__ =='__main__':
    for epoch in range(50000):#epoch 指的是所有样本训练一次，所以这是训练50000次，并且每2000次测试一下结果
        train(epoch)#这的train请看函数内详解
        if epoch%2000==1999:
            test()