import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

from MLP.mlp import mlp
from Data.csvDataLoader import prepare_data, my_collate_fn

# 检测设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dataset = prepare_data('./Data/test.csv')
test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False, collate_fn=my_collate_fn)

# 损失函数
criterion = nn.MSELoss()

# 测试mlp
model = torch.load("./savemodel/mlp.ckpt")
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    loss = 0
    for score, data in test_dataloader:
        RGB = data.to(torch.double).to(device)
        score = score.reshape(50, 1).to(device)
        outputs = model(RGB)
        print(outputs)
        for i in range(50):
            print(outputs[i], score[i])
            if (outputs[i] - score[i] <= 0.5) and (outputs[i] - score[i] >= -0.5):
                correct += 1
            total += 1
        loss += criterion(outputs, score)

    print(f"{total}: {correct}, \tloss: {loss}")
    print('Accuracy of four-layer-network on the 10000 test images: {} %'.format(100 * correct / total))