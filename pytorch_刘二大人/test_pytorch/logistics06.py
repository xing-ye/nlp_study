
import torch

import numpy as np
import matplotlib.pyplot as plt
x_data=torch.Tensor([[1.0],[2.0],[3.0]])#注意不能通小写tensor

y_data=torch.Tensor([[0],[0],[1]])

class LogisticModel(torch.nn.Module):
    def __init__(self):
        super(LogisticModel,self).__init__()
        self.linear=torch.nn.Linear(1,1)#输入输出维度

    def forward(self,x): #在call中被调用
        return torch.sigmoid(self.linear(x))#这里调用的是函数


model=LogisticModel()
criterion= torch.nn.BCELoss(reduction='sum' )#对损失求和返回标量
optimizer= torch.optim.SGD(model.parameters(),lr=0.01) #随机梯度下降

for epoch in range(1000):
    y_pred=model(x_data)

    loss=criterion(y_pred,y_data)

    #print(loss.item(),y_pred.data,y_pred.data.numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("w= ",model.linear.weight.item())
print("b= ",model.linear.bias.item())
#test
x=np.linspace(0,10,200)
x_t =torch.Tensor(x).view((200,1))#变形为200行1列的形式
y_t=model(x_t)
print(y_t)
print(y_t.data)
y=y_t.data.numpy()
print(y)
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='red')#这是画了一个(0,0.5)到(10,0.5)的直线，颜色color为红色
plt.xlabel('hours')
plt.ylabel('probability of pass')
plt.grid()#设置灰格子
plt.show()