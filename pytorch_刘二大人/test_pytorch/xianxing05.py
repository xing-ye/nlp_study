import torch

def compute(x):
    return w*x # 此处的w是Tensor类型，x由整型自动转为Tensor类型，并最终返回Tensor

def loss(x,y):
    return (compute(x)-y)**2

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w=torch.Tensor([1.0]) # 为Tensor赋初值时需要带方括号
w.requires_grad=True
# 标明需要计算该权值的梯度值，则计算图中，w参与运算的结果同样具有梯度值

for i in range(100):
    for x,y in zip(x_data,y_data):
        lossresult=loss(x,y)
        lossresult.backward()#反向传播，求出多个参数的梯度
        w.data-=0.01* w.grad.data#用w的梯度算
        w.grad.data.zero_()# 对于每个输入数据，使用后将w梯度置0，避免梯度累加
print('Time:' + str(i+1) + '\tw:' + str(w.data) + '\tresult:' + str(compute(4).data))
