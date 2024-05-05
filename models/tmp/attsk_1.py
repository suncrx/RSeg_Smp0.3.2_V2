#

import torch
from torch import nn
from functools import reduce


class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        采用分组卷积： groups = 32,所以输入channel的数值必须是group的整数倍
        '''
        super(SKConv,self).__init__()
        # 计算从向量C降维到 向量Z 的长度d
        d=max(in_channels//r,L)   
        self.M=M
        self.out_channels=out_channels
        
        # 根据分支数量 添加 不同核的卷积操作
        self.conv=nn.ModuleList()  
        for i in range(M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,
                                                     padding=1+i,dilation=1+i,
                                                     groups=32,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))
        # 自适应pool到指定维度    这里指定为1，实现 GAP            
        self.global_pool=nn.AdaptiveAvgPool2d(output_size = 1) 
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)  # 升维
        # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1
        self.softmax=nn.Softmax(dim=1) 
        
        
    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        #the part of split
        for i,conv in enumerate(self.conv):
            #print(i,conv(input).size())
            output.append(conv(input))    #[batch_size,out_channels,H,W]
        #the part of fusion
        # 逐元素相加生成 混合特征U  [batch_size,channel,H,W]
        U=reduce(lambda x,y:x+y,output) 
        #print(U.size())            
        s=self.global_pool(U)     # [batch_size,channel,1,1]
        #print(s.size())
        z=self.fc1(s)  # S->Z降维   # [batch_size,d,1,1]
        #print(z.size())
        #Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,
        #后一半为b   [batch_size,out_channels*M,1,1]
        a_b=self.fc2(z) 
        #print(a_b.size())
        #调整形状，变为 两个全连接层的值[batch_size,M,out_channels,1]  
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) 
        #print(a_b.size())
        # 使得两个全连接层对应位置进行softmax [batch_size,M,out_channels,1]  
        a_b=self.softmax(a_b) 
        #the part of selection
        #split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 
        #几个tensor块 [[batch_size,1,out_channels,1],[batch_size,1,out_channels,1]
        a_b=list(a_b.chunk(self.M,dim=1))
        #print(a_b[0].size())
        #print(a_b[1].size())
        # 将所有分块  调整形状，即扩展两维  
        # [[batch_size,out_channels,1,1],[batch_size,out_channels,1,1]
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) 
        # 权重与对应  不同卷积核输出的U 逐元素相乘[batch_size,out_channels,H,W] * 
        # [batch_size,out_channels,1,1] = [batch_size,out_channels,H,W]
        V=list(map(lambda x,y:x*y,output,a_b)) 
        # 两个加权后的特征 逐元素相加  [batch_size,out_channels,H,W] + 
        # [batch_size,out_channels,H,W] = [batch_size,out_channels,H,W]
        V=reduce(lambda x,y:x+y,V) 
        return V    # [batch_size,out_channels,H,W]
    
"""
x = torch.Tensor(8,32,24,24)
conv = SKConv(32,32,1,2,16,32)

print(conv(x).size())
"""



if __name__ == '__main__':
    
    sk64 = SKConv(64, 64, M=2, L=1, r=2)
                  
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', 
                           pretrained=True)
    layer1 = model.layer1 
    print(model.layer1)
    
    #add sk layers to layer1
    model.layer1 = nn.Sequential(layer1[0], sk64, 
                                 layer1[1], sk64)
    print(model.layer1)
        
    model.fc = nn.Linear(512, 6)
    print(model)
    
    x = torch.rand(4, 3,128,128)
    o = model(x)
    
    
    