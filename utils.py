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

def train(dataloader, model, loss_fn, optimiser, epoch):
    """training function"""

    num_batches = len(dataloader)
    model.train()
    train_loss, train_accuracy = 0, 0

    for X, y in dataloader:
        # Compute prediction error
        pred = model(X)
        # print(pred[:5])
        loss = loss_fn(pred, y)

        # Backpropagation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        train_loss += loss.item()
        train_accuracy += get_accuracy(pred, y)

    train_loss /= num_batches
    train_accuracy /= num_batches
    print(f"{epoch}. train loss: {train_loss}, train accuracy: ${train_accuracy}")
    return train_loss, train_accuracy.detach().numpy()

def test(dataloader, model, loss_fn, epoch):
    """testing function"""

    num_batches = len(dataloader)
    model.eval()
    test_loss, test_accuracy = 0, 0

    with torch.no_grad():

        for X, y in dataloader:

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            test_accuracy += get_accuracy(pred, y)

    test_loss /= num_batches
    test_accuracy /= num_batches
    print(f"{epoch}. test loss: {test_loss:>7f}, test accuracy: ${test_accuracy:,.2f}\n")
    return test_loss, test_accuracy.detach().numpy()

if __name__ == '__main__':
    max = 100
    pred = torch.tensor([[1],
                        [0.5],
                        [0.25]])

    target = torch.tensor([[0.5], 
                           [0.25],
                           [1]])

    print(get_accuracy(pred, target, max))