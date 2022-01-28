import torch

d_model=512 #embedding size词嵌入大小
d_ff=2048   # FeedForward dimension,全连接层维度
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder and Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
max_len=12
'''
位置编码
transpose,unsequenze的应用
unsequenze(i)加的维度是不是正好在i位置上啊？
https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
https://pytorch.org/docs/stable/generated/torch.nn.Module.html
'''
'''

pe=torch.zeros(max_len,d_model)#存储位置嵌入position_embedding
print(pe.shape)
position =torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
print(position.shape)
print(pe.unsqueeze(1).shape)
pe=pe.unsqueeze(0)
print(pe.shape)
pe=pe.transpose(0,1)
print(pe.shape)
'''
'''
torch.Size([12, 512])
torch.Size([12, 1])
torch.Size([12, 1, 512])
torch.Size([1, 12, 512])
torch.Size([12, 1, 512])
'''

'''
关于Linear的输入于输出
https://blog.csdn.net/qq_42079689/article/details/102873766
'''
'''
https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html
x = torch.tensor([1, 2, 3])
x.repeat(4, 2)
x.repeat(4, 2, 1).size()
'''
'''
tt3=torch.tensor([[-0.3623, -0.6115],
         [ 0.7283,  0.4699],
         [ 2.3261,  0.1599]])
result=tt3.view(-1)
print(result.shape)
'''
'''
x = torch.randn(2, 3)
print(x)
b=torch.cat((x, x, x), 0)
c=torch.cat((x, x, x), 1)
a=torch.cat((x,x),-1)
print(a)
'''
a = torch.rand((1,2,3))
print(a)
print(a.shape)
print(a.squeeze(1))
print(a.squeeze(1).shape)
