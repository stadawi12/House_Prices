import torch
import torch.nn as nn

class LinearModel(nn.Module):
    
    def __init__(self, input_size):
        super(LinearModel, self).__init__()

        self.model_size = 80

        self.linear1 = nn.Linear(input_size, self.model_size)
        self.linear2 = nn.Linear(self.model_size, self.model_size)
        self.activation = nn.ReLU()
        self.out = nn.Linear(self.model_size, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.out(x)
        return x

if __name__ == '__main__':
    
    x = torch.randn(1, 80)

    model = LinearModel(x.shape[1])

    out = model(x)
    print(out)
