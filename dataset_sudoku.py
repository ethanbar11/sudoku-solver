import numpy as np
from torch.utils.data import Dataset, DataLoader


class SudokuDataset(Dataset):
    def __init__(self, sudoku_path, mask_precentile=0.1, transform=None):
        self.sudoku_path = sudoku_path
        self.transform = transform
        self.mask_precentile = mask_precentile
        self.sudokus = np.load(sudoku_path)
        self.sudokus = self.sudokus.reshape(self.sudokus.shape[0], 81)
        self.sudokus = self.sudokus.astype(int)

    def __len__(self):
        return self.sudokus.shape[0]

    def __getitem__(self, idx):
        sudoku = self.sudokus[idx]
        sudoku = sudoku.reshape(81)
        sudoku = sudoku.astype(int)
        mask = np.random.choice(2, 81, p=[self.mask_precentile, 1 - self.mask_precentile])
        masked_sudoku = sudoku  # * mask
        return masked_sudoku, sudoku


if __name__ == '__main__':
    dataset = SudokuDataset(sudoku_path='scraper/sudokus.npy')
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)
    for i, (sudoku) in enumerate(dataloader):
        print(sudoku)
        break
