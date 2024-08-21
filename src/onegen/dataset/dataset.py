import torch

class BaseDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
    ):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError()
    
    def __len__(self):
        raise NotImplementedError()
    
    def read_db_file(self):
        raise NotImplementedError()

    def read_train_file(self):
        raise NotImplementedError()
