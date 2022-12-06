from torch.utils.data import Dataset

class AccidentSeverityDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.n_samples = len(x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples