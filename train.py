import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from MLP.mlp import mlp
from Data.csvDataLoader import prepare_data, my_collate_fn
from torch.utils.data import DataLoader

# 检测设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 3
hidden_size = 50
num_epochs = 1000
batch_size = 50  # 每一个batch的大小
learning_rate = 0.00005  # 学习率

train_dataset = prepare_data('./Data/train.csv')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)

# 构造mlp模型
model = mlp(input_size, hidden_size).to(device).double()

# 损失函数和优化函数
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_dataloader)
for epoch in range(num_epochs):
    model.train()
    for i, data in enumerate(train_dataloader):
        RGB = data[1].to(torch.double).to(device)
        score = data[0].to(torch.double).to(device)
        # outputs = []
        # for j in range(batch_size):
        #     outputs.append(model(RGB[j]))
        # outputs = torch.tensor(outputs).to(device)
        # print(outputs)
        outputs = model(RGB)
        # print(outputs)
        # print(score)
        loss = criterion(outputs, score)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

print(outputs)

torch.save(model, "./savemodel/mlp.ckpt")

test_dataset = prepare_data('./Data/test.csv')
test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False, collate_fn=my_collate_fn)

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    loss = 0
    for score, data in test_dataloader:
        RGB = data.to(torch.double).to(device)
        score = score.reshape(50, 1).to(device)
        outputs = model(RGB)
        for i in range(50):
            print(outputs[i], score[i])
            if (outputs[i] - score[i] <= 0.5) and (outputs[i] - score[i] >= -0.5):
                correct += 1
            total += 1
        loss += criterion(outputs, score)

    print(f"{total}: {correct}, \tloss: {loss}")
    print('Accuracy of four-layer-network on the 10000 test images: {} %'.format(100 * correct / total))