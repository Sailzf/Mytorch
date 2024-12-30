from torchvision import transforms, datasets
import cupy as cp
from time import time
import os
import nvtx
import swanlab
from PIL import Image
from tqdm import tqdm

from mytorch.ops import Max as mymax
from mytorch.tensor import Tensor, no_grad
from mytorch.dataset import Dataset
from mytorch.dataloader import DataLoader
import mytorch.module as nn
from mytorch.module import Module, Linear, Conv2D, MaxPooling2D, Dropout
import mytorch.functions as F
from mytorch.functions import relu, softmax
from mytorch.optim import Adam, SGD, Adagrad
from mytorch.loss import CrossEntropyLoss, NLLLoss
from mytorch import cuda

class ImageNetDataset(Dataset):
    def __init__(self, root, transform=None, is_train=True, cache_file=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.is_train = is_train
        self.cache_file = cache_file
        
        # 设置训练或验证目录
        self.data_dir = os.path.join(root, 'train' if is_train else 'val')
        
        print(f"Scanning {'training' if is_train else 'validation'} directory...")
        # 获取所有类别
        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # 获取所有图像文件路径和标签
        self.samples = []
        for class_name in tqdm(self.classes, desc="Scanning classes"):
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, img_name), class_idx))
        
        print(f"Found {len(self.samples)} images in {len(self.classes)} classes")
        
        # 如果存在缓存文件，加载缓存
        if cache_file and os.path.exists(cache_file + '_data.npy') and os.path.exists(cache_file + '_targets.npy'):
            print(f"Loading cached dataset from {cache_file}")
            with tqdm(total=2, desc="Loading cache") as pbar:
                self.cached_data = cp.load(cache_file + '_data.npy')
                pbar.update(1)
                self.cached_targets = cp.load(cache_file + '_targets.npy')
                pbar.update(1)
            self.use_cache = True
            print("Cache loaded successfully")
        else:
            self.use_cache = False
            if cache_file:
                print("Cache not found, will create cache during first epoch")
    
    def __getitem__(self, index):
        if self.use_cache:
            return Tensor(self.cached_data[index]), Tensor(self.cached_targets[index])
        
        path, target = self.samples[index]
        # 读取图像
        with Image.open(path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        
        # 转换为CuPy数组
        img_array = cp.array(img.numpy())
        target_array = cp.array(target)
        
        # 如果是第一次访问且需要缓存，开始创建缓存
        if not self.use_cache and self.cache_file and index == 0:
            self._start_caching()
        
        return Tensor(img_array), Tensor(target_array)
    
    def __len__(self):
        return len(self.samples)
    
    def _start_caching(self):
        print("\nStarting to cache dataset...")
        data = []
        targets = []
        
        # 使用tqdm创建进度条
        for path, target in tqdm(self.samples, desc="Caching images", unit="img"):
            with Image.open(path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
            data.append(cp.array(img.numpy()))
            targets.append(target)
        
        print("\nStacking arrays...")
        with tqdm(total=2, desc="Processing arrays") as pbar:
            data = cp.stack(data)
            pbar.update(1)
            targets = cp.array(targets)
            pbar.update(1)
        
        # 确保缓存目录存在
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        
        # 保存缓存
        print("\nSaving cache to disk...")
        with tqdm(total=2, desc="Saving cache") as pbar:
            cp.save(self.cache_file + '_data.npy', data)
            pbar.update(1)
            cp.save(self.cache_file + '_targets.npy', targets)
            pbar.update(1)
        
        print(f"Dataset cached successfully to {self.cache_file}")
        
        self.cached_data = data
        self.cached_targets = targets
        self.use_cache = True

# 初始化 SwanLab
run = swanlab.init(
    project="ImageNet-AlexNet",
    experiment_name="ImageNet-AlexNet-CuPy",
    config={
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 128,
        "num_epochs": 90,
        "device": "cuda" if cuda.is_available() else "cpu",
        "dropout_rate": 0.5,
        "weight_decay": 0.0005,
        "momentum": 0.9,
        "lr_scheduler": "step",
        "lr_step_size": 30,
        "lr_gamma": 0.1,
    },
)

# 数据预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
print("Loading datasets...")
train_dataset = ImageNetDataset(
    root='data/imagenet',
    transform=train_transform,
    is_train=True,
    cache_file='data/imagenet/cache/train'
)

val_dataset = ImageNetDataset(
    root='data/imagenet',
    transform=val_transform,
    is_train=False,
    cache_file='data/imagenet/cache/val'
)

train_loader = DataLoader(train_dataset, batch_size=run.config.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=run.config.batch_size, shuffle=False, num_workers=4)

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, dropout_rate=0.5):
        super(AlexNet, self).__init__()
        # 第一个卷积层块
        self.conv1 = Conv2D(3, 96, kernel_size=(11, 11), stride=4, padding=2)
        self.pool1 = MaxPooling2D(3, 3, 2)
        
        # 第二个卷积层块
        self.conv2 = Conv2D(96, 256, kernel_size=(5, 5), padding=2)
        self.pool2 = MaxPooling2D(3, 3, 2)
        
        # 第三个卷积层块
        self.conv3 = Conv2D(256, 384, kernel_size=(3, 3), padding=1)
        
        # 第四个卷积层块
        self.conv4 = Conv2D(384, 384, kernel_size=(3, 3), padding=1)
        
        # 第五个卷积层块
        self.conv5 = Conv2D(384, 256, kernel_size=(3, 3), padding=1)
        self.pool3 = MaxPooling2D(3, 3, 2)
        
        # 全连接层
        self.fc1 = Linear(256 * 6 * 6, 4096)
        self.fc2 = Linear(4096, 4096)
        self.fc3 = Linear(4096, num_classes)
        
        # Dropout层
        self.dropout = Dropout(p=dropout_rate)

    def forward(self, x):
        # 卷积层 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # 卷积层 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # 卷积层 3-5
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        
        # 展平
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x)

model = AlexNet(num_classes=1000, dropout_rate=run.config.dropout_rate)
device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")
model.to(device)

# 打印模型信息
total_params = sum(p.array().size for p in model.parameters())
print("\nModel Summary:")
print(f"Total Parameters: {total_params:,}")
print(f"Training Device: {device}")
print(f"Starting training for {run.config.num_epochs} epochs...")

criterion = NLLLoss()
optimizer = SGD(
    model.parameters(),
    lr=run.config.learning_rate,
    momentum=run.config.momentum,
    weight_decay=run.config.weight_decay
)

def adjust_learning_rate(epoch):
    """每30个epoch将学习率降低10倍"""
    lr = run.config.learning_rate * (0.1 ** (epoch // run.config.lr_step_size))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(epoch):
    with nvtx.annotate(f"Train Epoch {epoch}", color="green"):
        model.train()
        start_time = time()
        running_loss = 0.0
        total_batches = len(train_loader)
        correct = 0
        total = 0
        
        # 调整学习率
        current_lr = adjust_learning_rate(epoch)
        print(f"\nEpoch {epoch + 1}/{run.config.num_epochs}")
        print(f"Learning rate: {current_lr:.6f}")
        print(f"Training on {len(train_dataset)} samples with batch size {run.config.batch_size}")
        
        for batch_idx, (inputs, target) in enumerate(train_loader):
            with nvtx.annotate("Train Batch", color="lime"):
                progress = (batch_idx + 1) / total_batches * 100
                
                with nvtx.annotate("Forward Pass", color="cyan"):
                    outputs = model(inputs)
                    loss = criterion(outputs, target)
                
                with nvtx.annotate("Backward Pass", color="yellow"):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
                
                # 计算准确率
                predicted = mymax().forward(outputs.data, axis=1)
                total += target.array().size
                correct += (predicted == target.array()).sum().item()
                
                if batch_idx % 10 == 0:
                    avg_loss = running_loss / (batch_idx + 1)
                    accuracy = 100 * correct / total
                    
                    swanlab.log({
                        "train/loss": avg_loss,
                        "train/accuracy": accuracy,
                    }, step=epoch * total_batches + batch_idx)
                    
                    print(f"\rProgress: [{batch_idx:>4d}/{total_batches:>4d}] {progress:>3.0f}% "
                          f"Loss: {avg_loss:.4f} Acc: {accuracy:.2f}%", end="")
        
        # 打印epoch总结
        epoch_loss = running_loss / total_batches
        epoch_accuracy = 100 * correct / total
        epoch_time = time() - start_time
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Average Loss: {epoch_loss:.4f}")
        print(f"Accuracy: {epoch_accuracy:.2f}%")
        print(f"Time Used: {epoch_time:.2f} seconds")
        print(f"Samples/second: {len(train_dataset) / epoch_time:.2f}")

        swanlab.log({
            "train/epoch_loss": epoch_loss,
            "train/epoch_accuracy": epoch_accuracy,
            "train/epoch_time": epoch_time,
            "train/samples_per_second": len(train_dataset) / epoch_time,
            "train/learning_rate": current_lr
        }, step=epoch)

def validate(epoch):
    with nvtx.annotate("Validation", color="red"):
        model.eval()
        start_time = time()
        total_loss = 0
        correct = 0
        total = 0
        total_batches = len(val_loader)
        
        print("\nEvaluating on validation set...")
        
        with no_grad():
            for batch_idx, (inputs, target) in enumerate(val_loader):
                with nvtx.annotate("Val Batch", color="pink"):
                    progress = (batch_idx + 1) / total_batches * 100
                    
                    with nvtx.annotate("Forward Pass", color="orange"):
                        outputs = model(inputs)
                        loss = criterion(outputs, target)
                        predicted = mymax().forward(outputs.data, axis=1)
                    
                    total_loss += loss.item()
                    total += target.array().size
                    correct += (predicted == target.array()).sum().item()
                    
                    print(f"\rProgress: [{batch_idx:>4d}/{total_batches:>4d}] {progress:>3.0f}%", end="")
        
        # 计算最终指标
        avg_loss = total_loss / total_batches
        accuracy = 100 * correct / total
        val_time = time() - start_time
        
        # 记录到SwanLab
        swanlab.log({
            "val/loss": avg_loss,
            "val/accuracy": accuracy,
            "val/time": val_time
        }, step=epoch)
        
        # 打印验证结果总结
        print(f"\nValidation Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Time Used: {val_time:.2f} seconds")
        print(f"Samples/second: {len(val_dataset) / val_time:.2f}")

if __name__ == '__main__':
    with nvtx.annotate("Training Loop", color="blue"):
        for epoch in range(run.config.num_epochs):
            train(epoch)
            validate(epoch) 