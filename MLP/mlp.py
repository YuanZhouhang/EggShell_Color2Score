import torch.nn as nn


# MLP网络
class mlp(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到影藏层
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)  # 影藏层到输出层
        self.dropout = nn.Dropout(p=0.5)  # dropout训练

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
