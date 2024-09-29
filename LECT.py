import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from utils.lossFunctions import *

#1.可变形卷积（Deformable Convolution）
class DeformConv(nn.Module):
    #inc：输入通道数；outc：输出通道数；kernel_size：卷积核大小；padding：填充大小；stride：步长；modulation：调制
    #调制操作用于在可变形卷积的基础上引入可学习的权重调整
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv, self).__init__()
        self.kernel_size = kernel_size#卷积核心大小
        self.padding = padding#填充
        self.stride = stride#步长
        self.zero_padding = nn.ZeroPad2d(padding)#零填充
        # conv则是最终实际要进行的卷积操作，注意这里步长设置为卷积核大小
        # 因为与该卷积核进行卷积操作的特征图是由输出特征图中每个点扩展为其对应卷积核那么多个点后生成的
        #比如conv是3x3卷积，输出特征图尺寸2x3（hxw），那么其每个点都被扩展为9（3x3）个点，对应卷积核的每个位置。
        # 于是与conv进行卷积的特征图尺寸变为(2x3) x (3x3)，将stride设置为3，最终输出的特征图尺寸就刚好是2x3。
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        #p_conv：偏置层，是生成offsets所使用的卷积，输出通道数为卷积核尺寸的平方的2倍，代表对应卷积核每个位置横纵坐标都有偏移量。
        self.p_conv = nn.Conv2d(in_channels, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)#将 self.p_conv 层的权重初始化为常数值 0。
        #这行代码注册了一个反向传播钩子（backward hook）到 self.p_conv 层上。
        # 反向传播钩子是在反向传播过程中自动执行的函数，可以用来对梯度进行操作。在这里，它调用了静态方法 _set_lr，将梯度乘以 0.1。
        self.p_conv.register_full_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
        #m_conv：权重学习层，kernel_size*kernel_size代表了卷积核每个元素的权重
            self.m_conv = nn.Conv2d(in_channels, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

#这是一个类内部的静态方法，用于设置反向传播过程中梯度的学习率。在这里，它将所有梯度乘以 0.1。
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

#前向传播
    def forward(self, x):
        #使用 self.p_conv 对输入 x 进行卷积操作，得到偏移量 offset。这个偏移量将用于在输入特征图上进行采样，使卷积核的位置变得可变。
        offset = self.p_conv(x)#(batch_size, 2*N, height, width)
        #如果modulation设置为true的话就在可变形卷积的基础上引入可学习的权重调整
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2#N 表示每个像素点在特征图上的两个方向（x 和 y）上的偏移量数

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w):通过self._get_p()计算在输入特征图上的采样位置p,获取所有卷积核的中心坐标
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)：对采样位置 p 进行调整和剪裁，以确保它们在合理的范围内。
        p = p.contiguous().permute(0, 2, 3, 1)
        #双线性插值：将某个浮点坐标的左上、左下、右上、右下四个位置的像素值按照与该点的距离计算加权和，作为该点处的像素值。
        # lt, rb, lb, rt 分别代表左上，右下，左下，右上。
        q_lt = p.detach().floor()#取整
        q_rb = q_lt + 1#在 q_lt 的基础上，每个元素都增加 1，用于计算采样位置的右下角。
        #对 q_lt、q_rb 进行调整和剪裁，以确保它们的坐标在合理的范围内（0 到特征图尺寸的边界）
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p，对 p 进行调整和剪裁
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)，计算权重系数
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)：根据坐标去取到对应位置的像素值
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)，根据采样位置和权重系数，计算特征图上采样点的偏移特征 x_offset。
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation，如果有权重的话，要计算出 m，乘到 x_offset 上。
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        #(B,c,h,w,N) to (B,c,h,w)
        x_offset = self._reshape_x_offset(x_offset, ks)
        #送入 self.conv 进行卷积计算
        out = self.conv(x_offset)
        return out

 #_get_p_n()返回了一个卷积核内部的坐标网格，以卷积核中心p_0为原点。
    # 如果是 3×3 卷积核的话，横纵坐标刻度应该都是 [-1, 0, 1]，这是相对卷积核中心的相对坐标。
    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),indexing='ij')
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    #获得卷积核的中心坐标
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride),indexing='ij')
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    #根据计算出的位置坐标，得到该位置的像素值
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    #我们在取完像素值之后得到的 x_offset 的形状是 b , c , h , w , N 而我们要的形状肯定是 b , c , h , w。因此这里还有一个 reshape 的操作，

    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks ,w*ks)

        return x_offset

#2.深度可分离卷积
class DWConv(nn.Module):
    #主要分为两个过程，分别为逐通道卷积（Depthwise Convolution）和逐点卷积（Pointwise Convolution）
    def __init__(self, in_channels, out_channels):
        super(DWConv, self).__init__()  # super：调用 DWConv 类的父类，即 nn.Module
        #分组卷积层
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth_conv = nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=3, stride=1, padding=1)
        #逐点卷积层
        self.point_conv = nn.Conv2d(in_channels, out_channels, groups=1, kernel_size=1, stride=1, padding=0)

    #前向传播
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

#3.局部增强型MLP（LEMLP）
class LEMlp(nn.Module):
    def __init__(self, in_channels,out_channels, act_layer=nn.GELU, drop=0.1):
        super(LEMlp, self).__init__()
        self.in_channels = in_channels
        self.expansion = 4
        hidden_channels = max(self.in_channels,out_channels)//self.expansion
        #线性层，激活函数层
        self.linear1 = nn.Sequential(nn.Conv2d(in_channels, hidden_channels,kernel_size=1,stride=1,padding=0),
                                     act_layer())
        #深度可分离卷积层，激活函数层
        self.dwconv = nn.Sequential(DWConv(hidden_channels,hidden_channels),
                                    nn.GroupNorm(hidden_channels//4,hidden_channels),
                                    act_layer())
        #线性层
        self.linear2 = nn.Sequential(nn.Conv2d(hidden_channels, out_channels,kernel_size=1,stride=1,padding=0))
        #drop层
        self.drop = nn.Dropout(drop)

    #前向传播
    def forward(self, x):
        x = self.linear1(x) #线性激活
        x = self.drop(x)    #dropout
        x = self.dwconv(x)  #深度可分离卷积，激活
        x = self.linear2(x) #线性激活
        x = self.drop(x)#dropout
        return x

#4.Self-attention （注意力机制）
class SA(nn.Module):
    #in_channels：输入通道数，out_channels：输出通道数，heads：注意力头数，k：查询和键的通道数，u 是值的通道数，m 是局部注意力的窗口大小(卷积核大小）。
    def __init__(self, in_channels, out_channels, heads=4, k=16, u=1, m=7):
        super(SA, self).__init__()
        #self.kk:查询和键的通道数,self.uu:值的通道数，self.vv：每个头部输出的通道数，self.mm：局部注意力的窗口大小，self.heads：注意力头数。
        self.kk, self.uu, self.vv, self.mm, self.heads = k, u, out_channels // heads, m, heads
        # self.padding：计算局部注意力的填充大小。
        self.padding = (m - 1) // 2

        #self.queries：用于生成查询向量（queries）
        self.queries = nn.Sequential(
            #卷积层，输入通道数为in_channels，输出通道数为k * heads，使用 1x1 的卷积核进行卷积操作，无偏置项
            nn.Conv2d(in_channels, k * heads, kernel_size=1, bias=False),
            #分组归一化层，对输入进行归一化操作。参数 k*heads//4 表示分组数
            nn.GroupNorm(k*heads//4,k*heads),
        )
        #self.keys：用于生成键向量（keys）
        self.keys = nn.Sequential(
            #卷积层，输入通道数为in_channels，输出通道数为k * u，使用 1x1 的卷积核进行卷积操作，无偏置项
            nn.Conv2d(in_channels, k * u, kernel_size=1, bias=False),
            # 分组归一化层，对输入进行归一化操作。参数 k*u//4 表示分组数
            nn.GroupNorm(k*u//4,k*u)
        )
        #self.values：用于生成值向量（values）
        self.values = nn.Sequential(
            #卷积层，输入通道数为in_channels，输出通道数为self.vv* u，使用 1x1 的卷积核进行卷积操作，无偏置项
            nn.Conv2d(in_channels, self.vv * u, kernel_size=1, bias=False),
            # 分组归一化层，对输入进行归一化操作。参数vv*u//4表示分组数
            nn.GroupNorm(self.vv*u//4,self.vv*u)
        )

        # softmax 函数，用于在最后一个维度上进行 softmax 操作，以获取注意力权重。
        self.softmax = nn.Softmax(dim=-1)
        #论文中relative position embedding模块，用于计算局部注意力操作中的输入序列元素之间的相对位置关系，这个嵌入矩阵是一个可训练的参数
        #requires_grad=True：它会在模型的训练过程中被优化，self.kk：查询和键的通道数，self.uu：值的通道数，1：输入数据的特征图数目，m：局部注意力窗口的大小
        self.embedding = nn.Parameter(torch.randn([self.kk, self.uu, 1, m, m]), requires_grad=True)

    # 定义局部注意力机制的 forward（前向传播） 方法
    def forward(self, x):
        n_batch, C, w, h = x.size()#获取输入张量x的维度，n_batch：批量处理的样本数量（即批量大小），C通道维数，w，h：宽度，高度
        #输入张量x，输出（queries）张量，n_batch: 批量大小，self.heads: 注意力头的数量，self.kk：查询和键的通道数，w * h: 输入张量的宽度和高度的乘积，用于展平查询张量。
        queries = self.queries(x).view(n_batch, self.heads, self.kk, w * h)
        #输入输入张量x，输出（keys）张量，self.kk：查询和键的通道数，self.uu：值的通道数，然后沿着最后一个维度应用softmax函数
        softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, w * h))
        #输入张量x，输出（values）张量，self.vv：每个头部输出的通道数，self.uu：值的通道数
        values = self.values(x).view(n_batch, self.vv, self.uu, w * h)
        #经过softmax转换的键(softmax)和值(values)进行矩阵乘法运算，用于计算查询（queries）与键（keys）之间的相似度，并根据相似度加权求和对值进行聚合。
        #K：查询和键的通道数，v：每个头部输出的通道数
        content = torch.einsum('bkun,bvun->bkv', (softmax, values))
        #h：注意力头的数量，v：每个头部输出的通道数，n：w * h，content表示最终的基于内容的注意力。
        content = torch.einsum('bhkn,bkv->bhvn', (queries, content))
        #这行代码将值(values)张量的形状调整，对比(n_batch, self.vv, self.uu, w * h)
        values = values.view(n_batch, self.uu, -1, w, h)#b,u,v,w,h
        #将 values 张量与 self.embedding 权重进行卷积，计算上下文注意力
        # 填充(padding)被应用以确保输出具有与输入相同的空间维度。
        context = F.conv3d(values, self.embedding, padding=(0, self.padding, self.padding))
        #将张量 context 的形状调整为 (n_batch, self.kk, self.vv, w * h)。
        context = context.view(n_batch, self.kk, self.vv, w * h)
        #将queries张量和context张量按照指定的维度标记进行乘法运算
        #上下文注意力和基于内容的注意力输出均为(n_batch, self.heads, self.vv, w * h)
        context = torch.einsum('bhkn,bkvn->bhvn', (queries, context))
        #将内容注意力 (content) 和上下文注意力 (context) 相加，得到一个新的张量 out。
        out = content + context
        #对张量 out 进行了连续化 (contiguous) 操作，并通过调用 view 方法将其形状重新调整为 (n_batch, -1, w, h)。
        #其中 -1 表示该维度的大小会根据其他维度的大小自动确定，以保持原始张量的元素总数不变。
        out = out.contiguous().view(n_batch, -1, w, h)
        #返回最终的总的注意力
        return out

#5.局部增强型注意力层（LE Self attention layer）
class LESAlayer(nn.Module):
    #in_chnnels：输入通道数，out_chnnels：输出通道数，drop_path=0.1：使用随机深度，act_layer=nn.GELU：GELU激活函数
    #drop=0.1: 是指在局部增强型多层感知机（LEMLP）模块中，应用于隐藏层的丢弃率（dropout rate）
    def __init__(self, in_channels, out_channels,drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.GroupNorm):
        super(LESAlayer, self).__init__()#调用父类
        #将输入通道数和输出通道数保存为实例变量
        self.in_chnnels = in_channels
        self.out_chnnels = out_channels
        #创建一个 LocalSABlock 的实例,该模块用于执行局部自注意力操作，并将输入通道数和输出通道数作为参数传递给它。
        self.attens = SA(
            in_channels=in_channels, out_channels=out_channels
        )
        #层归一化
        self.norm1 = norm_layer(out_channels//4,out_channels)
        #局部增强模块（使用DWConv）
        self.lem = nn.Sequential(
            DWConv(in_channels, out_channels),
            nn.GroupNorm(out_channels//4,out_channels),
            act_layer()
        )
        # 创建一个 DropPath 的实例，用于实现随机深度。如果 drop_path 大于 0，则使用 DropPath 模块，否则使用恒等映射 nn.Identity()。
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #创建一个 Mlp 的实例，传递输入通道数，输出通道数，GELU激活函数，辍学率，linear进行初始化参数。该模块包含多层感知机（MLP）操作。
        self.mlp = LEMlp(out_channels, out_channels, act_layer=act_layer, drop=drop)
        #局部增强模块
        self.norm2 = norm_layer(out_channels//4,out_channels)

    def forward(self, x):
        lem_x = self.lem(x)  # 对输入特征进行深度可分离卷积，以增强局部特征
        sa_x  = self.norm1(self.attens(x))#输入特征通过注意力机制
        x1 = x+ self.drop_path(sa_x)+lem_x #表示将自注意力机制的结果与原始输入进行残差连接。
        x =  x1 + self.drop_path(self.norm2((self.mlp(x1))))  # 经过MLP残差连接
        return x#返回最终的结果


#6.通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels,act_layer=nn.GELU):
        super(ChannelAttention, self).__init__()
        self.expansion = 4
        # 局部增强模块（使用DWConv）
        self.lem = nn.Sequential(
            DWConv(in_channels, out_channels),
            nn.GroupNorm(out_channels//4,out_channels),
            act_layer()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.fc = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // self.expansion,kernel_size=1,stride=1,padding=0),
            act_layer(),
            nn.Conv2d(out_channels // self.expansion, out_channels,kernel_size=1,stride=1,padding=0),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lem_x = self.lem(x)
        avg_out = self.avg_pool(lem_x)
        max_out = self.max_pool(lem_x)

        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out)
        x = out * lem_x

        return x

#7.Bottleneck残差块结构，旨在解决深层网络训练中的梯度消失和参数过多的问题。它在残差连接中引入了一个降维层和一个升维层，通过使用较少的参数来提高网络的表达能力。
#其中包括了卷积、归一化、非线性变换以及残差连接等操作，从而实现了特征的提取和转换。
class Bottleneck(nn.Module):
    #in_planes：输入通道数，planes：输出通道数，stride=1：步长
    def __init__(self, in_channels, out_channels, stride=1,act_layer=nn.GELU):
        super(Bottleneck, self).__init__()
        #设置了扩展系数 self.expansion 为 4，用于定义 Bottleneck 模块中通道数的变化比例。
        self.expansion = 4
        #然后计算了隐藏通道数 hidden_planes，它是输入通道数和输出通道数中较大值除以扩展系数的结果。
        hidden_channels = max(in_channels,out_channels) // self.expansion
        #设置一个1x1的卷积层，用于将输入通道数降维为隐藏层通道数。bias=False：无偏置层
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        #设置一个Group Normalization层，将隐藏层通道数hidden_planes分成（hidden_planes //4）个分组进行特征归一化。
        self.bn1 = nn.GroupNorm(hidden_channels //4,
                                hidden_channels)
        # 通道注意力
        self.lem1 = nn.Sequential(
            ChannelAttention(in_channels, hidden_channels),
        )
        #设置一个模块列表，其中包含一个TFBlock模块，输入和输出通道数都为隐藏层通道数。
        self.conv2 = nn.ModuleList([LESAlayer(hidden_channels, hidden_channels)])
        #设置另一个Group Normalization层，将隐藏层通道数hidden_planes分成（hidden_planes //4）个分组进行特征归一化。
        self.bn2 = nn.GroupNorm(hidden_channels // 4,
                                hidden_channels)
        #将GELU激活函数添加到模块列表的末尾
        self.conv2.append(nn.GELU())
        #将模块列表转换为序列模块，以便可以按顺序应用其中的模块。
        self.conv2 = nn.Sequential(*self.conv2)
        #定义另一个1x1的卷积层，用于将隐藏层通道数恢复为输出通道数。
        self.conv3 = nn.Conv2d(hidden_channels,  out_channels, kernel_size=1, bias=False)
        # 设置第三个Group Normalization层，将输出通道数planes分成（planes // 4）个分组进行特征归一化。
        self.bn3 = nn.GroupNorm(out_channels // 4,out_channels)
        # 通道注意力
        self.lem2 = nn.Sequential(
            ChannelAttention(hidden_channels, out_channels),
        )
        #设置一个GELU激活函数，用于非线性变换。
        self.GELU=nn.GELU()
        #定义一个空的Sequential模块，用于表示残差连接的捷径路径。
        self.shortcut = nn.Sequential()
        # 局部增强模块
        self.lem3 = nn.Sequential(
            DWConv(in_channels, out_channels),
            nn.GroupNorm(out_channels//4,out_channels),
            act_layer()
        )


    #定义残差块结构的前向传播方法
    def forward(self, x):
        #将输入x经过卷积层self.conv1，然后通过归一化层self.bn1进行特征归一化，最后通过GELU激活函数self.GELU进行非线性变换，得到x1。
        x1 = self.GELU(self.bn1(self.conv1(x)))
        #输入x经过局部增强模块，得到lem1
        lem1 = self.lem1(x)
        #级联输出
        x_hidden=x1+lem1
        #将x输入到LE self attention layer
        x_hidden = self.conv2(x_hidden)
        #同上
        x2 = self.GELU(self.bn3(self.conv3(x_hidden)))
        lem2 = self.lem2(x_hidden)
        x_lem=self.lem3(x)
        x = x_lem+x2+lem2
        #返回最终的输出x
        return x


#8.LESABlock模块：实现局部增强型SA模块（LE Self attention block)
class LESABlock(nn.Module):
    def __init__(self, in_channels, out_channels):#输入输出通道数
        super().__init__()
        self.SABlock = Bottleneck(in_channels, out_channels)
        self.activation=torch.nn.GELU()
    #Trans_EB模块的前向传播方法
    def forward(self, x):
        x = self.SABlock(x)
        x = self.activation(x)
        return x#返回最终的输出X

#9.下采样模块（使用3*3 Conv strid=2）
class Merge_Block(nn.Module):
    def __init__(self,in_channels,out_channels, norm_layer=nn.GroupNorm):
        #dim输入维度，dim_out输出维度， resolution输入图像的分辨率。
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        #卷积降维
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1)#3x3 的卷积核，步幅为 2，填充为 1，从而将特征图的尺寸减半。
        #归一化层
        self.norm = norm_layer(out_channels//4,out_channels)
        #激活函数层
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))#对调整后的特征图进行卷积、归一化和激活函数处理，实现降采样操作。
        return x

#10.#上采样模块
class UP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)#3x3 的卷积层 self.conv1
        self.bn1 = nn.GroupNorm(in_channels//4,in_channels)#对卷积后的特征图进行归一化处理。
        self.act1 = nn.GELU()#激活函数层
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn2 = nn.GroupNorm(out_channels//4,out_channels)
        self.act2 = nn.GELU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.act1(self.bn1(x))
        #进行上采样操作，将特征图的尺寸扩大为原来的两倍。这里使用双线性插值法进行上采样。
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv2(x)
        x = self.act2(self.bn2(x))
        return x

#11.卷积主干模块
class DCONVM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = DeformConv(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn2 = nn.GroupNorm(out_channels//4,out_channels)
        self.act2 = nn.GELU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.act1(self.bn1(x))
        x = self.conv2(x)
        x = self.act2(self.bn2(x))
        return x

#12.下采样模块(Encoder)
class Down1(nn.Module):
    def __init__(self):
        super(Down1, self).__init__()
        #特征通道数3——64
        self.nn1 = DCONVM(3,64)
        self.nn2 = LESABlock(64, 64)
        #特征图512-256
        self.nn3 = Merge_Block(64,64)#特征图减半

    #Down1模块的前向传播方法：
    def forward(self, inputs):
        scale1_1 = self.nn1(inputs)
        scale1_2 = self.nn2(scale1_1)
        scale1_3 = self.nn3(scale1_2)

        return scale1_1,scale1_2,scale1_3

class Down2(nn.Module):
    def __init__(self):
        super(Down2,self).__init__()
        ##特征通道64-128
        self.nn1 = LESABlock(64,128)
        self.nn2 = LESABlock(128,128)
        # 特征图256-128
        self.nn3 = Merge_Block(128, 128)  # 特征图减半

    # Down2模块的前向传播方法：
    def forward(self, inputs):
        scale2_1 = self.nn1(inputs)
        scale2_2 = self.nn2(scale2_1)
        scale2_3 = self.nn3(scale2_2)
        return scale2_1,scale2_2,scale2_3

class Down3(nn.Module):
    def __init__(self):
        super(Down3,self).__init__()
        ##特征通道128-256
        self.nn1 = LESABlock(128,256)
        self.nn2 = LESABlock(256,256)
        self.nn3 = LESABlock(256,256)
        # 特征图128-64
        self.nn4 = Merge_Block(256, 256)  # 特征图减半

    # Down3模块的前向传播方法：
    def forward(self, inputs):
        scale3_1 = self.nn1(inputs)
        scale3_2 = self.nn2(scale3_1)
        scale3_3 = self.nn3(scale3_2)
        scale3_4 = self.nn4(scale3_3)
        return scale3_1, scale3_2, scale3_3, scale3_4

class Down4(nn.Module):
    def __init__(self):
        super(Down4,self).__init__()
        ##特征通道256-512
        self.nn1 = LESABlock(256,512)
        self.nn2 = LESABlock(512,512)
        self.nn3 = LESABlock(512,512)

    # Down4模块的前向传播方法：
    def forward(self, inputs):
        scale4_1 = self.nn1(inputs)
        scale4_2 = self.nn2(scale4_1)
        scale4_3 = self.nn3(scale4_2)
        return scale4_1, scale4_2, scale4_3



class Fuse1(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.convblock = nn.Conv2d(in_channels, out_channels, 1 ,padding=0)# 1x1 的卷积层 self.convblock，用于将输入的两个特征图进行通道维度上的融合。
        self.bn1 = nn.GroupNorm(out_channels//4,out_channels)#用于对融合后的特征图进行归一化处理。
        self.act1 = nn.GELU()#激活归一化后的特征图。

    def forward(self, up_feature):
        #上采样的特征）按照通道维度进行拼接，拼接后的特征 x 用于进行特征融合。这里假设 up_feature 是一个列表，包含多个上采样后的特征图
        x = torch.cat(up_feature, dim=1)
        #断言语句，用于检查拼接后的特征 x 的通道数是否与 ConvBlock 的输入通道数相同，确保特征融合操作的正确性。
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.act1(self.bn1(feature))  #对特征进行归一化和非线性激活操作。
        return x

#特征融合和分类预测
class Fuse2(nn.Module):
# scale:上采样的尺度因子，用于控制上采样后的特征图尺寸相对于输入特征图的缩放比例。num_class: 分类器输出的类别数量。
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels, 3, padding=1)#用于将上采样后的特征图进行特征融合，将通道数从 in_ 调整为 out。
        self.activation = nn.GELU()#激活卷积后的特征图。
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)#用于将特征融合后的特征图映射为分类器输出所需的类别数量

    def forward(self, up_inp):
        #输入的上采样特征 up_inp 进行上采样操作，将特征图的尺寸按照 self.scale 的缩放因子进行放大，得到上采样后的特征图 outputs。
        outputs = self.conv1(up_inp)
        outputs = self.activation(outputs)
        outputs = self.conv2(outputs)
        return outputs

#14.最终特征融合模块：用于将多个不同来源的特征图进行融合，并最终输出用于分类的特征图。
class FeatureFusionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_class):
        super().__init__()
        self.in_channels = in_channels
        self.convblock = nn.Conv2d(in_channels, out_channels, 3 ,padding=1)# 3x3 的卷积层 self.convblock，用于将输入的两个特征图进行通道维度上的融合。
        self.bn1 = nn.BatchNorm2d(out_channels)#用于对融合后的特征图进行归一化处理。
        self.act1 = nn.GELU()#激活归一化后的特征图。
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))#定义自适应平均池化层 self.avgpool，用于对输入特征图进行全局平均池化，得到全局平均特征。
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)#定义一个 1x1 的卷积层 self.conv1，用于对全局平均特征进行通道维度上的变换。
        self.act2 = nn.GELU()#用于激活卷积后的特征图。
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()#用于将卷积后的特征图的值限制在 [0, 1] 范围内。
        self.conv_last = DCONVM(out_channels, out_channels)
        #输出的类别数量，用于分类任务中的类别预测。这里将输出特征图的通道数转换为预测的类别数量。
        self.conv_last2 = nn.Conv2d(out_channels, num_class, kernel_size=1, bias=False)

    def forward(self, up_feature):
        #上采样的特征）按照通道维度进行拼接，拼接后的特征 x 用于进行特征融合。这里假设 up_feature 是一个列表，包含多个上采样后的特征图
        x = torch.cat(up_feature, dim=1)
        #断言语句，用于检查拼接后的特征 x 的通道数是否与 ConvBlock 的输入通道数相同，确保特征融合操作的正确性。
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        feature = self.act1(self.bn1(feature))  #对特征进行归一化和非线性激活操作。
        x = self.avgpool(x)
        x = self.act2(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        #将特征融合后的结果 feature 与经过特征重要性控制的全局平均特征 x 进行按元素相乘，得到最终的融合特征。
        x = torch.mul(feature, x)
        #将融合后的特征 x 与特征 feature 进行按元素相加，用于保留原始特征的信息
        x = torch.add(x, feature)
        #对上采样后的特征图进行一个 3x3 的卷积操作，用于进一步的特征变换。
        x = self.conv_last(x)
        #对上采样后的特征图进行一个 1x1 的卷积操作，得到最终输出的特征图用于分类任务。
        x = self.conv_last2(x)

        return x


#15.LECrackformer整体网络结构
class LECT(nn.Module):
    def __init__(self):
        super(LECT, self).__init__()

        #Encoder
        self.down1 = Down1()
        self.down2 = Down2()
        self.down3 = Down3()
        self.down4 = Down4()

        # Decoder
        self.decoder4_1 = UP(64 * 8,64 * 4)
        self.decoder4_2 = UP(64 * 4,64 * 2)
        self.decoder4_3 = UP(64 * 2,64)

        self.ffm3_1 = Fuse1(64 * 8,64 * 4)
        self.ffm2_1 = Fuse1(64 * 4,64 * 2)
        self.ffm1_1 = Fuse1(64 * 2 ,64)

        self.decoder3_1 = UP(64 * 4,64 * 2)
        self.decoder3_2 = UP(64 * 2,64)

        self.decoder2_1 = UP(64 * 2,64)

        self.ffm_final = FeatureFusionModule(64 * 4,64,1)

        self.aux1 = Fuse2(64, 1)
        self.aux2 = Fuse2(64, 1)
        self.aux3 = Fuse2(64, 1)
        self.aux4 = Fuse2(64, 1)
        # 这行代码调用了一个私有函数_init_weights，它用于初始化模型的权重。这个函数的具体实现应该在后面的代码中可以找到。
        self.apply(self._init_weights)


    # init_weights，用于初始化模型的权重,它接受一个模块 m 作为输入参数。
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):  # 模块若为线性层（nn.Linear 类型）
            trunc_normal_(m.weight, std=.02)  # 则使用 trunc_normal_ 函数对权重进行截断正态分布初始化，标准差为 0.02。
            if isinstance(m, nn.Linear) and m.bias is not None:  # 再次判断模块是否为线性层，并且是否具有偏置项
                nn.init.constant_(m.bias, 0)  # 如果是，则使用 nn.init.constant_将偏置项初始化为常数 0。
        elif isinstance(m, nn.Conv2d):  # 如果模块是卷积层（nn.Conv2d 类型），
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # 计算 fan_out，即输出通道数乘以卷积核大小的乘积，
            fan_out //= m.groups  # 将fan_out（连接到下一层的权重的输出通道数）进行除法运算并向下取整，以适应分组卷积的情况。
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))  # 对卷积层的权重进行初始化，使用了正态分布初始化方法，均值为 0，标准差
            if m.bias is not None:  # 如果卷积层有偏置项 (m.bias)
                m.bias.data.zero_()  # 则将偏置项初始化为零 (m.bias.data.zero_())。

    def calculate_loss(self, outputs, labels):
        loss = 0
        loss = cross_entropy_loss_RCF(outputs, labels)
        return loss

 #定义网络整体架构前向传播过程
    def forward(self, inputs):
        # Encoder部分：通过调用down1到down5模块进行特征提取和下采样，得到各个尺度的特征scale1_1到scale5_3。

        #scale1_3:(_,64,512,512),scale1_4:(_,64,256,256)
        scale1_1,scale1_2,scale1_3 = self.down1(inputs)
        #scale2_3:(_,128,256,256),scale2_4:(_,128,128,128)
        scale2_1,scale2_2,scale2_3 = self.down2(scale1_3)
        #scale3_3:(_,256,128,128),scale3_4:(_,256,64,64)
        scale3_1,scale3_2,scale3_3,scale3_4 = self.down3(scale2_3)
        #scale4_3:(_,512,64,64),scale4_4:(_,512,32,32)
        scale4_1,scale4_2,scale4_3 = self.down4(scale3_4)


        #Decoder部分：
        # scale4_1:(_,256,128,128)
        x4_1 = self.decoder4_1(scale4_3)
        # x4_1:(_,128,256,256)
        x4_2 = self.decoder4_2(x4_1)
        #x4_2:(_,64,512,512)
        x4_3= self.decoder4_3(x4_2)

        # ffm4_1:(_,256,128,128)
        ffm3_1 = self.ffm3_1([x4_1,scale3_3])
        # ffm3_1:(_,128,256,256)
        ffm2_1 = self.ffm2_1([x4_2,scale2_2])
        # ffm2_1:(_,64,512,512)
        ffm1_1 = self.ffm1_1([x4_3,scale1_2])

        # x3_1:(_,128,256,256)
        x3_1 = self.decoder3_1(ffm3_1)
        # x3_2:(_,64,512,512)
        x3_2 = self.decoder3_2(x3_1)

        # x2_1:(_,64,512,512)
        x2_1 = self.decoder2_1(ffm2_1)

        #
        aux1 = self.aux1(ffm1_1)
        aux2 = self.aux2(x2_1)
        aux3 = self.aux3(x3_2)
        aux4 = self.aux4(x4_3)


        #(_,64,512,512)
        output = self.ffm_final([ffm1_1,x3_2,x2_1,x4_3])

        return aux1,aux2,aux3,aux4,output

# 实例化你的模型
model = LECT()

# 计算参数数量
total_params = sum(p.numel() for p in model.parameters())
total_params_million = total_params / 1e6

print(f"总参数量：{total_params}，大约 {total_params_million:.2f} M")

if __name__ == '__main__':
    #这里假设输入是一个大小为512x512的RGB图像，批次大小为1
    inp = torch.randn(1, 3, 512, 512)
    model = LECT()#创建了 crackformer 的一个实例，即实例化了模型对象。
    out=model(inp)#将输入张量 inp 传递给模型 model 进行前向传播，得到输出张量 out。
    print(model)#print(model) 打印模型的结构信息，包括模型的各个组件和参数





