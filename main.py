import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import dataset_sudoku
from FFN import NeuralNetwork


def train(dataset, model, loss_fn, optimizer, device, epochs=1):
    print("Training...")
    print('Model : {}'.format(model))
    print('Loss function : {}'.format(loss_fn))
    print('Optimizer : {}'.format(optimizer))
    size = len(dataset)
    train_percentage = 0.8
    RANDOM_SEED = 42
    BATCH_SIZE = 100

    torch.manual_seed(RANDOM_SEED)
    train_size = int(train_percentage * size)
    test_size = size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.train()

    for epoch in range(epochs):
        print('Epoch : {}/{}'.format(epoch+1,epochs))
        avg_acc = []
        total_loss = []
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            X = convert_to_one_hot(X)

            # Compute prediction error
            pred = model(X)
            y = convert_to_one_hot(y)
            avg_acc.append(((pred.argmax(dim=2) == y.argmax(dim=2)).long().sum(dim=1) / X.shape[1]).mean().item())
            loss = loss_fn(pred, y)
            total_loss.append(loss.item())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 200 == 0:
                current = batch * len(X)
                acc = avg_acc[-1]  # np.array(avg_acc).mean()
                printable_loss = np.array(total_loss).mean()
                print(f"loss: {printable_loss:>7f}  [{current:>5d}/{size:>5d}] , avg_acc: {acc * 100:>7f}%")
        acc = np.array(avg_acc).mean()
        printable_loss = np.array(total_loss).mean()
        print(f"Train total loss: {printable_loss:>7f}  , avg_acc: {acc * 100:>7f}%")
        test(test_loader, model, loss_fn, device)


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()
    loss = []
    avg_acc = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            # TODO: maybe remove this line
            X = convert_to_one_hot(X)
            pred = model(X)
            y = convert_to_one_hot(y)
            avg_acc.append(((pred.argmax(dim=2) == y.argmax(dim=2)).long().sum(dim=1) / X.shape[1]).mean().item())
            loss.append(loss_fn(pred, y).item())
    print(f"Validation loss: {np.array(loss).mean():>7f} , avg_acc: {np.array(avg_acc).mean() * 100:>7f}%")


def convert_to_one_hot(y):
    # TODO: The output of the network is a vector of length 9 containing numbers from 0 to 8.
    # Thus, we decrease the number by one to get number in [0, 8] and then we convert it to one-hot encoding.
    y_new = y.cpu().detach().numpy() - 1
    a = (np.arange(y_new.max() + 1) == y_new[..., None]).astype(int)
    return torch.from_numpy(a).float()


# This is the main file for the project.
# It is responsible for the main loop of the program.
# I will use it for training the model and evaluation later on.
if __name__ == '__main__':
    # Game hyperparameters
    MASK_PRECENTAGE = 0.1

    # Training hyperparameters
    EPOCHS = 100
    lr = 3e-3
    ########### Model hyperparameters
    HIDDEN_SIZE = 256
    N_LAYERS = 8
    # HIDDEN_LAYERS = [HIDDEN_SIZE for _ in range(N_LAYERS)]
    # HIDDEN_LAYERS = [256, 128, 256, 128, 256, 128, 256, 128]
    HIDDEN_LAYERS = [81 * 9]

    ########### end of hyperparameters
    sudoku_path = './/scraper//sudokus.npy'
    dataset = dataset_sudoku.SudokuDataset(sudoku_path, MASK_PRECENTAGE)
    model = NeuralNetwork(HIDDEN_LAYERS)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(dataset, model, loss, optimizer, device, epochs=EPOCHS)
