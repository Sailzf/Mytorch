from flask import Flask, render_template, jsonify, request, send_file
import threading
import os
import sys
import graphviz
import time
import logging
from openai import OpenAI
import swanlab
import cupy as cp

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))  # 修正根目录路径
sys.path.append(root_dir)

from mytorch.dataloader import prepare_mnist_data, DataLoader
from mytorch.loss import NLLLoss
from mytorch.optim import Adam, SGD, Adagrad
from mytorch.module import Conv2D, Linear, MaxPooling2D, Module
import mytorch.functions as F
from mytorch import cuda

# 初始化 SwanLab
run = swanlab.init(
    logdir='./logs',
    mode="local",
    project="MNIST-LeNet",
    experiment_name="MNIST-LeNet-Web",
    config={
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "batch_size": 64,
        "num_epochs": 10,
        "device": "cuda" if cuda.is_available() else "cpu",
    },
)

# 设置数据目录
data_dir = os.path.join(root_dir, 'data')

# 准备数据集
train_dataset = prepare_mnist_data(root=data_dir, backend='cupy', train=True)
test_dataset = prepare_mnist_data(root=data_dir, backend='cupy', train=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")

# 设置模板和静态文件夹的绝对路径
template_dir = os.path.join(current_dir, 'templates')
static_dir = os.path.join(current_dir, 'static')

# 确保目录存在
os.makedirs(template_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

app = Flask(
    __name__,
    template_folder=template_dir,
    static_folder=static_dir
)

# 全局变量
training_thread = None
is_training = False
model = None
current_epoch = 0
current_batch = 0
total_batches = 0
samples_processed = 0
latest_accuracy = 0.0
latest_loss = 0.0
training_start_time = 0
training_logs = []

# 训练参数
training_params = {
    'learning_rate': 0.01,
    'batch_size': 64,
    'epochs': 10,
    'optimizer': 'Adam',
    'log_interval': 300  # 每多少个batch输出一次日志
}

# 网络结构参数
network_params = {
    'conv1_out_channels': 6,
    'conv1_kernel_size': 5,
    'conv2_out_channels': 16,
    'conv2_kernel_size': 5,
    'fc1_out_features': 120,
    'fc2_out_features': 84,
    'pool_size': 2,
    'pool_stride': 2
}

# 设置Flask日志级别
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # 只显示错误日志

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-uGNIeQaOCBkeqsuZfrN691FmRym1KqxUhjSPWDfjZOMXerro"),
    base_url="https://api.chatanywhere.tech/v1"
)

class CustomLeNet(Module):
    def __init__(self, params):
        super(CustomLeNet, self).__init__()
        # 第一个卷积层
        self.conv1 = Conv2D(
            1,
            params['conv1_out_channels'],
            (params['conv1_kernel_size'], params['conv1_kernel_size'])
        )
        # 第一个池化层
        self.pool1 = MaxPooling2D(
            params['pool_size'],
            params['pool_stride'],
            params['pool_size']
        )
        # 第二个卷积层
        self.conv2 = Conv2D(
            params['conv1_out_channels'],
            params['conv2_out_channels'],
            (params['conv2_kernel_size'], params['conv2_kernel_size'])
        )
        # 第二个池化层
        self.pool2 = MaxPooling2D(
            params['pool_size'],
            params['pool_stride'],
            params['pool_size']
        )
        
        # 计算全连接层的输入特征数
        # 28x28 -> conv1 -> 24x24 -> pool1 -> 12x12 -> conv2 -> 8x8 -> pool2 -> 4x4
        fc1_in_features = params['conv2_out_channels'] * 4 * 4
        
        # 三个全连接层
        self.fc1 = Linear(fc1_in_features, params['fc1_out_features'])
        self.fc2 = Linear(params['fc1_out_features'], params['fc2_out_features'])
        self.fc3 = Linear(params['fc2_out_features'], 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x

def generate_network_graph():
    """生成网络结构图"""
    try:
        dot = graphviz.Digraph(comment='LeNet Architecture')
        dot.attr(rankdir='LR')  # 从左到右布局
        
        # 设置全局图形属性
        dot.attr('node', style='filled', fillcolor='lightblue')
        
        # 计算特征图大小
        input_size = 28
        conv1_size = input_size - network_params['conv1_kernel_size'] + 1
        pool1_size = conv1_size // network_params['pool_size']
        conv2_size = pool1_size - network_params['conv2_kernel_size'] + 1
        pool2_size = conv2_size // network_params['pool_size']
        
        # 添加节点
        dot.node('input', f'Input\n1x{input_size}x{input_size}', shape='box', fillcolor='lightgreen')
        dot.node('conv1', f'Conv1\n{network_params["conv1_out_channels"]}@{conv1_size}x{conv1_size}\n{network_params["conv1_kernel_size"]}x{network_params["conv1_kernel_size"]} kernel', shape='box')
        dot.node('pool1', f'Pool1\n{network_params["conv1_out_channels"]}@{pool1_size}x{pool1_size}\n{network_params["pool_size"]}x{network_params["pool_size"]} kernel', shape='box', fillcolor='lightyellow')
        dot.node('conv2', f'Conv2\n{network_params["conv2_out_channels"]}@{conv2_size}x{conv2_size}\n{network_params["conv2_kernel_size"]}x{network_params["conv2_kernel_size"]} kernel', shape='box')
        dot.node('pool2', f'Pool2\n{network_params["conv2_out_channels"]}@{pool2_size}x{pool2_size}\n{network_params["pool_size"]}x{network_params["pool_size"]} kernel', shape='box', fillcolor='lightyellow')
        
        fc1_in = network_params["conv2_out_channels"] * pool2_size * pool2_size
        dot.node('fc1', f'FC1\n{fc1_in}→{network_params["fc1_out_features"]}', shape='box', fillcolor='lightpink')
        dot.node('fc2', f'FC2\n{network_params["fc1_out_features"]}→{network_params["fc2_out_features"]}', shape='box', fillcolor='lightpink')
        dot.node('fc3', f'FC3\n{network_params["fc2_out_features"]}→10', shape='box', fillcolor='lightpink')
        dot.node('output', 'Output\n10', shape='box', fillcolor='lightgreen')
        
        # 添加边
        dot.edge('input', 'conv1', 'Conv')
        dot.edge('conv1', 'pool1', 'MaxPool')
        dot.edge('pool1', 'conv2', 'Conv')
        dot.edge('conv2', 'pool2', 'MaxPool')
        dot.edge('pool2', 'fc1', 'Flatten+FC')
        dot.edge('fc1', 'fc2', 'FC')
        dot.edge('fc2', 'fc3', 'FC')
        dot.edge('fc3', 'output', 'LogSoftmax')
        
        # 保存图片
        output_file = os.path.join(static_dir, 'network_structure')
        dot.render(output_file, format='png', cleanup=True)
        print(f"网络结构图已生成: {output_file}.png")
        return True
    except Exception as e:
        print(f"生成网络结构图时出错: {str(e)}")
        return False

def get_optimizer(name, parameters, lr):
    optimizers = {
        'Adam': Adam,
        'SGD': SGD,
        'Adagrad': Adagrad
    }
    return optimizers[name](parameters, lr=lr)

def initialize_model():
    global model
    model = CustomLeNet(network_params)
    model.to(device)
    # 生成网络结构图
    if not generate_network_graph():
        print("警告：网络结构图生成失败")
    return model

def train_epoch(epoch, model, optimizer, criterion):
    global current_batch, samples_processed, latest_loss
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    print(f"\n【Epoch {epoch + 1}】")
    epoch_start_time = time.time()  # 开始计时
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if not is_training:
            break
            
        current_batch = batch_idx + 1
        samples_processed += len(data)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # 计算训练准确率
        pred = output.data.argmax(axis=1)
        correct += (pred == target.data).sum().item()
        total += target.data.shape[0]
        
        # 更新最新损失
        latest_loss = loss.item()
        running_loss += latest_loss
        
        # 根据设置的间隔打印进度
        if batch_idx % training_params['log_interval'] == 0:
            accuracy = 100. * correct / total if total > 0 else 0
            avg_loss = running_loss / (batch_idx + 1)
            print(f"Batch [{batch_idx + 1}/{len(train_loader)}]  - Loss: [{latest_loss:.4f}]  - Accuracy: [{accuracy:.2f}%]", end="\r")
            
            # 记录到SwanLab
            run.log({
                "main/loss": avg_loss,
                # "train/batch_loss": latest_loss,
                # "train/batch_accuracy": accuracy,
            }, step=epoch * len(train_loader) + batch_idx)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_time = time.time() - epoch_start_time  # 计算耗时
    swanlab.log({
        "train/epoch_loss": epoch_loss,
        "train/epoch_time": epoch_time,
        "main/samples_per_second": len(train_loader.dataset) / epoch_time
    }, step=epoch)
    print(f"\nTime Used: {epoch_time:.2f} s")
    return epoch_loss

def test_epoch(model, criterion):
    global latest_accuracy, latest_loss
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    print(f"\n【Test {current_epoch + 1}】")
    test_start_time = time.time()  # 开始计时
    
    with cp.cuda.Device(0):
        for batch_idx, (data, target) in enumerate(test_loader):
            if not is_training:
                break
                
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.argmax(axis=1)
            correct += (pred == target.data).sum().item()
            total += target.data.shape[0]  # 使用shape[0]获取批次大小
            
            # 打印进度条
            print(f"Batch [{batch_idx + 1}/{len(test_loader)}]  - Accuracy: [{100. * correct / total:.2f}%]", end="\r")
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total if total > 0 else 0
    
    # 更新最新指标
    latest_accuracy = accuracy
    latest_loss = test_loss
    
    test_time = time.time() - test_start_time  # 计算耗时
    swanlab.log({
        "main/accuracy": accuracy,
        "test/loss": test_loss,
        "test/time": test_time
    }, step=current_epoch)
    
    print(f"\nTime Used: {test_time:.2f} s")
    
    return accuracy, test_loss

def training_process():
    global is_training, current_epoch, current_batch, samples_processed, latest_accuracy, latest_loss
    global training_start_time, model, training_params, total_batches
    
    optimizer = get_optimizer(
        training_params['optimizer'],
        model.parameters(),
        training_params['learning_rate']
    )
    criterion = NLLLoss()
    
    training_start_time = time.time()
    current_batch = 0
    samples_processed = 0
    total_batches = len(train_loader)
    
    while is_training and current_epoch < training_params['epochs']:
        epoch_start_time = time.time()
        
        # 训练一个epoch
        train_loss = train_epoch(current_epoch, model, optimizer, criterion)
        
        # 如果训练被中断，退出循环
        if not is_training:
            break
            
        # 测试当前模型
        accuracy, test_loss = test_epoch(model, criterion)
        
        current_epoch += 1
        
        # 添加训练日志
        training_logs.append({
            'epoch': current_epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'accuracy': accuracy,
            'time': time.time() - epoch_start_time,
            'status': 'completed'
        })

@app.route('/')
def home():
    return render_template('train.html')

@app.route('/get_params')
def get_params():
    return jsonify(training_params)

@app.route('/update_params', methods=['POST'])
def update_params():
    global training_params
    if is_training:
        return jsonify({'status': 'error', 'message': '训练进行中，无法更新参数'})
    
    data = request.get_json()
    
    # 验证参数
    required_params = ['learning_rate', 'batch_size', 'epochs', 'optimizer', 'log_interval']
    if not all(key in data for key in required_params):
        return jsonify({'status': 'error', 'message': '参数不完整'})
    
    # 验证参数范围
    if not (0 < data['learning_rate'] <= 1):
        return jsonify({'status': 'error', 'message': '学习率必须在0-1之间'})
    if not (1 <= data['batch_size'] <= 512):
        return jsonify({'status': 'error', 'message': 'Batch Size必须在1-512之间'})
    if not (1 <= data['epochs'] <= 100):
        return jsonify({'status': 'error', 'message': '训练轮数必须在1-100之间'})
    if not (1 <= data['log_interval'] <= 60000):
        return jsonify({'status': 'error', 'message': '日志输出间隔必须在1-6000 ）之间'})
    if data['optimizer'] not in ['Adam', 'SGD', 'Adagrad']:
        return jsonify({'status': 'error', 'message': '不支持的优化器类型'})
    
    # 更新参数
    training_params.update(data)
    return jsonify({'status': 'success', 'message': '参数更新成功'})

@app.route('/start_training')
def start_training():
    global training_thread, is_training, model, current_epoch
    
    if not is_training:
        is_training = True
        current_epoch = 0  # 重置epoch计数
        if model is None:
            model = initialize_model()
        training_thread = threading.Thread(target=training_process)
        training_thread.start()
        return jsonify({'status': 'success', 'message': '训练已开始'})
    return jsonify({'status': 'error', 'message': '训练已在进行中'})

@app.route('/stop_training')
def stop_training():
    global is_training
    is_training = False
    return jsonify({'status': 'success', 'message': '训练已停止'})

@app.route('/get_status')
def get_status():
    global training_start_time, samples_processed, current_batch, total_batches
    
    # 计算训练速度（样本/秒）
    training_speed = 0
    elapsed_time = 0
    if training_start_time > 0:
        elapsed_time = time.time() - training_start_time
        if elapsed_time > 0:
            training_speed = samples_processed / elapsed_time
    
    # 计算总批次数
    if total_batches == 0 and train_loader is not None:
        total_batches = len(train_loader)
    
    # 格式化数值
    formatted_speed = f"{training_speed:.1f}"
    formatted_accuracy = f"{latest_accuracy:.2f}%" if latest_accuracy > 0 else "0.00%"
    formatted_loss = f"{latest_loss:.4f}" if latest_loss > 0 else "0.0000"
    formatted_time = f"{elapsed_time:.1f}"
    
    return jsonify({
        'is_training': is_training,
        'current_epoch': current_epoch,
        'total_epochs': training_params['epochs'],
        'current_batch': current_batch,
        'total_batches': total_batches,
        'samples_processed': samples_processed,
        'total_samples': len(train_loader.dataset) if train_loader else 0,
        'latest_accuracy': formatted_accuracy,
        'latest_loss': formatted_loss,
        'training_speed': formatted_speed,
        'elapsed_time': formatted_time,
        'batch_size': training_params['batch_size'],
        'logs': training_logs[-10:] if training_logs else []  # 只返回最近10条日志
    })

@app.route('/network_structure')
def get_network_structure():
    """获取网络结构图"""
    try:
        image_path = os.path.join(static_dir, 'network_structure.png')
        if not os.path.exists(image_path):
            if not generate_network_graph():
                return "无法生成网络结构图", 500
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        print(f"访问网络结构图时出错: {str(e)}")
        return str(e), 500

@app.route('/get_network_params')
def get_network_params():
    return jsonify(network_params)

@app.route('/update_network_params', methods=['POST'])
def update_network_params():
    global network_params, model
    if is_training:
        return jsonify({'status': 'error', 'message': '训练进行中，无法更新网络结构'})
    
    data = request.get_json()
    
    # 验证参数
    required_params = [
        'conv1_out_channels', 'conv1_kernel_size',
        'conv2_out_channels', 'conv2_kernel_size',
        'fc1_out_features', 'fc2_out_features',
        'pool_size', 'pool_stride'
    ]
    
    if not all(key in data for key in required_params):
        return jsonify({'status': 'error', 'message': '网络参数不完整'})
    
    # 验证参数范围
    if not all(isinstance(data[key], int) and data[key] > 0 for key in required_params):
        return jsonify({'status': 'error', 'message': '所有参数必须为正整数'})
    
    # 验证卷积核大小
    if data['conv1_kernel_size'] >= 28 or data['conv2_kernel_size'] >= 12:
        return jsonify({'status': 'error', 'message': '卷积核大小过大'})
    
    # 更新参数
    network_params.update(data)
    
    # 重新初始化模型
    model = None
    
    # 重新生成网络结构图
    if not generate_network_graph():
        return jsonify({'status': 'error', 'message': '网络结构图生成失败'})
    
    return jsonify({'status': 'success', 'message': '网络结构更新成功'})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '')
        
        # 使用最近的训练日志作为上下文
        recent_logs = training_logs[-5:] if training_logs else []
        log_context = "\n".join([
            f"Epoch {log['epoch']}: Accuracy {log['accuracy']:.2f}%, Loss {log['test_loss']:.4f}"
            for log in recent_logs
        ])
        
        context = f"""
最近的训练日志:
{log_context}

用户问题: {user_message}
"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个深度学习专家，专门帮助用户分析MNIST数据集上的LeNet模型训练情况。请根据提供的训练日志，为用户提供专业的分析和建议。"},
                {"role": "user", "content": context}
            ]
        )
        
        return jsonify({
            'status': 'success',
            'response': response.choices[0].message.content
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print(f"【训练控制面板启动成功！】开始训练请访问: http://127.0.0.1:5000")
    
    # 确保启动时生成网络结构图
    if not os.path.exists(os.path.join(static_dir, 'network_structure.png')):
        if not generate_network_graph():
            print("警告：启动时网络结构图生成失败")
    
    app.run(host='0.0.0.0', port=5000, debug=False) 