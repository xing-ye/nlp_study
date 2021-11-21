import numpy as np
import torch
import matplotlib.pyplot as plt

xy=np.loadtxt('diabetes.csv.gz',delimiter=',',dtype=np.float32)#可以用来读取txt和csv
print(xy)
x_data=torch.from_numpy(xy[:,:-1])#-1代表最后一列，所以由于左闭右开，实际上到了倒数第二列
y_data=torch.from_numpy(xy[:,[-1]])#若提示没有包是因为缺少pylint依赖


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

epoch_list=[]
loss_list=[]

for epoch in range(1000):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    epoch_list.append(epoch)
    loss_list.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(epoch_list,loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')

plt.show()
layer1_weight=model.linear1.weight.data
layer1_bias=model.linear1.bias.data
print(layer1_weight)
print(layer1_bias)

