import torch
from torchvision import transforms #注意不是transformer
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional  as F
import torch.optim as optim

batch_size=64

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.137,),(0.3081,))
])
'''
用来将原始的图片转换为tensor张量
（将输入的数据shape W，H，C ——> C，W，H）c代表通道数（比如灰色只有一个通道，而带颜色的可能会有红黄绿三通道）
同时要将矩阵中的值（像素）从0~256映射到0~1区间上
最后将数据集转换为正态分布，
因为神经网络对正态分布的学习效果最好。第一个为均值，第二个为标准差（不要少了逗号），这些数据需要提前计算
本次的数据来自于经验（因为是很常用的数据集）
这实际上就是标准化的操作（变成了0-1分布，即标准正态分布）
'''
train_dataset=datasets.MNIST(
    root='./dataset/mnist',#数据集存放位置
    train=True,#是否是训练集
    download=False,#是否需要下载
    transform=transform,#数据操作，张量和正太分布
)
train_loader=DataLoader(
    dataset=train_dataset,
    shuffle=True,#是否打乱
    batch_size=batch_size,#minibatch大小
)#加载数据

test_dataset=datasets.MNIST(
    root='./dataset/mnist',#数据集存放位置
    train=False,#是否是训练集
    download=False,#是否需要下载
    transform=transform,#数据操作，张量和正态分布
)
test_loader=DataLoader(
    dataset=test_dataset,
    shuffle=False,#是否打乱
    batch_size=batch_size,#minibatch大小
)#加载数据

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1=torch.nn.Linear(784,512)#输入是784
        self.l2=torch.nn.Linear(512,256)
        self.l3=torch.nn.Linear(256,128)
        self.l4=torch.nn.Linear(128,64)
        self.l5=torch.nn.Linear(64,10)#最后的输出为10（0~9共10类别）
        #神经网络的层数也是一种超参数（需手动设置的），会影响训练效果

    def forward(self,x):
        x=x.view(-1,784)
        #minist的数据是一个28X28的矩阵（28*28=784）
        # 由于全连接神经网络需要输入一阶向量,所以将矩阵变成一个向量
        #-1代表会自动计算出minibatch_size，
        # 输入是一个（N，1，28，28），
        # 其中N一个minibatch中的样本数，那么会算出-1=N，行数，也就是样本数
        x=F.relu(self.l1(x))#选用relu作为激活函数，
        # 激活函数的作用是改变之前数据的线性关系，防止多层神经网络转换成一层的（无效化了）
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=F.relu(self.l4(x))
        x=self.l5(x)
        #最后的输出不需要进行激活，而直接计算交叉熵，
        # 因为交叉熵函数包括了softmax层
        return x

model=Net()
criterion=torch.nn.CrossEntropyLoss()
#交叉熵损失函数最后会对softmax的输出做log并且乘以一个one-hot向量
#这样只会得到概率最大的概率作为y_hat，可以看原博文
optimzer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

'''
上面momentum为动量因子，其作用是提高收敛速度，增加稳定性而且还有摆脱局部最优的能力
若当前梯度的方向与历史梯度一致（表明当前样本不太可能为异常点），则会增强这个方向的梯度，
若当前梯度与历史梯方向不一致，则梯度会衰减
'''
def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):#这部分的说明可以看前面写过的
        inputs,labels=data
        #这里的实现在Dataset的__getitem__中实现，可以看第七节的代码dataload7.py
        optimzer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,labels)

        loss.backward()
        optimzer.step()
        running_loss+=loss.item()

        if batch_idx%300==299:#每300次输出一下平均损失
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss=0.0


def test():
    correct,total=0,0
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            outputs=model(images)
            _,predicted=torch.max(outputs.data,dim=1)
            #计算每一行的最大概率值的(每一行就是每一个样本)，即求每个样本的最大概率类别
            #同时会返回结果和下标，这里我们只需要下标即可，_只是用来接受结果，但不会用到
            total+=labels.size(0)#label.size(0算的是每一个minibatch的样本数（行数）)
            correct+=(predicted==labels).sum().item()
            #求出所有预测正确的数量(若不同为0相同为1，自然会求出正确的结果)
        print('accuracy on test set: %d %% ' % (100 * correct / total))

if __name__=='__main__':
    for epoch in range(10):
        train(epoch)
        test()

