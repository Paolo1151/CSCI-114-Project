from torch.utils.data import Dataset, DataLoader

class AccidentSeverityAutoencoderDataset(Dataset):
    def __init__(self, x):
        self.x = x
        
        self.n_samples = len(x)

    def __getitem__(self, index):
        return self.x[index], self.x[index]

    def __len__(self):
        return self.n_samples