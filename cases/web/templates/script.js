function updateStatus() {
    fetch('/get_status')
        .then(response => response.json())
        .then(data => {
            const status = data.is_training ? '训练中' : '已停止';
            document.getElementById('trainingStatus').textContent = status;
            document.getElementById('currentEpoch').textContent = data.current_epoch;
            document.getElementById('totalEpochs').textContent = data.total_epochs;
            document.getElementById('currentBatch').textContent = data.current_batch;
            document.getElementById('totalBatches').textContent = Math.ceil(60000 / data.batch_size); // MNIST训练集大小
            document.getElementById('samplesProcessed').textContent = data.samples_processed;
            document.getElementById('trainingSpeed').textContent = data.training_speed;
            document.getElementById('latestAccuracy').textContent = data.latest_accuracy;
            document.getElementById('latestLoss').textContent = data.latest_loss;
            document.getElementById('elapsedTime').textContent = data.elapsed_time + ' 秒';
            
            // 更新进度条
            const progress = (data.current_epoch / data.total_epochs) * 100;
            document.getElementById('progressBar').style.width = `${progress}%`;
        });
}

// 保存网络结构参数
document.getElementById('saveNetworkParamsBtn').addEventListener('click', () => {
    const params = {
        conv1_out_channels: parseInt(document.getElementById('conv1_out_channels').value),
        conv1_kernel_size: parseInt(document.getElementById('conv1_kernel_size').value),
        conv2_out_channels: parseInt(document.getElementById('conv2_out_channels').value),
        conv2_kernel_size: parseInt(document.getElementById('conv2_kernel_size').value),
        fc1_out_features: parseInt(document.getElementById('fc1_out_features').value),
        fc2_out_features: parseInt(document.getElementById('fc2_out_features').value),
        pool_size: parseInt(document.getElementById('pool_size').value),
        pool_stride: parseInt(document.getElementById('pool_stride').value)
    };

    fetch('/update_network_params', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(params)
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        if (data.status === 'success') {
            // 刷新网络结构图
            const img = document.getElementById('networkStructureImg');
            img.src = '/network_structure?' + new Date().getTime();
        }
    });
});

// 保存训练参数
document.getElementById('saveParamsBtn').addEventListener('click', () => {
    const params = {
        learning_rate: parseFloat(document.getElementById('learning_rate').value),
        batch_size: parseInt(document.getElementById('batch_size').value),
        epochs: parseInt(document.getElementById('epochs').value),
        optimizer: document.getElementById('optimizer').value,
        log_interval: parseInt(document.getElementById('log_interval').value)
    };

    fetch('/update_params', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(params)
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
    });
});

document.getElementById('startBtn').addEventListener('click', () => {
    fetch('/start_training')
        .then(response => response.json())
        .then(data => {
            alert(data.message);
            updateStatus();
        });
});

document.getElementById('stopBtn').addEventListener('click', () => {
    fetch('/stop_training')
        .then(response => response.json())
        .then(data => {
            alert(data.message);
            updateStatus();
        });
});

// 每秒更新状态
setInterval(updateStatus, 1000);

// 页面加载时获取当前参数
Promise.all([
    fetch('/get_params').then(response => response.json()),
    fetch('/get_network_params').then(response => response.json())
]).then(([trainParams, networkParams]) => {
    // 设置训练参数
    document.getElementById('learning_rate').value = trainParams.learning_rate;
    document.getElementById('batch_size').value = trainParams.batch_size;
    document.getElementById('epochs').value = trainParams.epochs;
    document.getElementById('optimizer').value = trainParams.optimizer;
    document.getElementById('log_interval').value = trainParams.log_interval;

    // 设置网络参数
    document.getElementById('conv1_out_channels').value = networkParams.conv1_out_channels;
    document.getElementById('conv1_kernel_size').value = networkParams.conv1_kernel_size;
    document.getElementById('conv2_out_channels').value = networkParams.conv2_out_channels;
    document.getElementById('conv2_kernel_size').value = networkParams.conv2_kernel_size;
    document.getElementById('fc1_out_features').value = networkParams.fc1_out_features;
    document.getElementById('fc2_out_features').value = networkParams.fc2_out_features;
    document.getElementById('pool_size').value = networkParams.pool_size;
    document.getElementById('pool_stride').value = networkParams.pool_stride;
});

// 发送消息函数
function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    if (!message) return;
    
    // 禁用输入和发送按钮
    input.disabled = true;
    document.getElementById('sendButton').disabled = true;
    
    // 添加用户消息
    addMessage(message, 'user');
    
    // 清空输入框
    input.value = '';
    
    // 发送到服务器
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            addMessage(data.response, 'ai');
        } else {
            addMessage('抱歉，出现了一些问题：' + data.message, 'ai');
        }
    })
    .catch(error => {
        addMessage('抱歉，请求失败：' + error.message, 'ai');
    })
    .finally(() => {
        // 重新启用输入和发送按钮
        input.disabled = false;
        document.getElementById('sendButton').disabled = false;
        input.focus();
    });
}

// 添加消息到聊天窗口
function addMessage(message, type) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const messageContent = document.createElement('div');
    messageContent.textContent = message;
    messageDiv.appendChild(messageContent);
    
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString();
    messageDiv.appendChild(timeDiv);
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// 在Enter键按下时发送消息
document.getElementById('chatInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// 设置预设问题
function setPresetQuestion(question) {
    document.getElementById('chatInput').value = question;
    sendMessage();
}