import torch
import yaml

def Read_Input(input_path):
    with open(input_path, 'r') as input_file:
        input_data = yaml.load(input_file, Loader=yaml.FullLoader)
    return input_data

def get_accuracy(pred, target):
    """Calculates accuracy of prediction"""
    difference = abs(pred - target)
    return torch.mean(difference)

if __name__ == '__main__':
    max = 100
    pred = torch.tensor([[1],
                        [0.5],
                        [0.25]])

    target = torch.tensor([[0.5], 
                           [0.25],
                           [1]])

    print(get_accuracy(pred, target, max))
