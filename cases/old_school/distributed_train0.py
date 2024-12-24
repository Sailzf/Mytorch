import argparse
from mytorch.distributed import RingAllReduce
from mytorch.tensor import Tensor
from mytorch.optim import Adam
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int)
    parser.add_argument('--world-size', type=int)
    args = parser.parse_args()
    
    # 节点地址配置（实际使用时应从配置文件读取）
    nodes = [
        ('113.54.241.168', 9000),
        ('113.54.254.115', 9001),
        ('113.54.254.115', 9002)
    ]
    
    # 初始化Ring-AllReduce
    distributed = RingAllReduce(args.rank, args.world_size, nodes)
    
    '''
    # 创建模型和优化器
    model = YourModel()
    optimizer = Adam(model.parameters())

    # 训练循环
    for epoch in range(num_epochs):
        for batch in dataloader:  # 假设数据已经按rank划分好
            # 前向传播
            output = model(batch.data)
            loss = criterion(output, batch.target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 同步梯度
            for param in model.parameters():
                if param.grad is not None:
                    # 使用Ring-AllReduce同步梯度
                    param.grad = distributed.allreduce(param.grad)
                    # 因为是累加，所以需要除以进程数
                    param.grad.data /= args.world_size
            
            # 更新参数
            optimizer.step()
            
            if args.rank == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
    '''
    tensor = np.random.rand(6, 6)
    print(f"original tensor: {tensor}")

    result = distributed.allreduce(tensor)


if __name__ == '__main__':
    main()