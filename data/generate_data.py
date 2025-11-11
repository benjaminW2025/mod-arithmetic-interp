import torch
from torch.utils.data import Dataset

class ModularArithmeticDataset(Dataset):
    '''
    Class which generates the data for modular arithemetic training
    '''
    def __init__(self, modulus=113, operation='add'):
        self.modulus = modulus
        self.operation = operation

        # generate all pairs of data
        self.pairs = [(a, b) for a in range(modulus) for b in range(modulus)]

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__ (self, idx):
        # gets the data point
        a, b = self.pairs[idx]

        if (self.operation == 'add'):
            result = (a + b) % self.modulus
        # may add more operations in the future

        # return the two operands and their residue
        return a, b, result