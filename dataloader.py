import random
import numpy as np

class DataLoader():
    def __init__(self, dataset, shuffle = False, batch_size = 1):
        self.shuffle = shuffle
        self.batch_size = batch_size
        
        if shuffle:
            random.shuffle(dataset)

        if batch_size > len(dataset):
            raise ValueError('批次数量不能大于数据数量')
        
        split_size = len(dataset) / batch_size
        self.dataset = np.array_split(dataset, split_size)

        self.current_index = -1
        self.max_index = len(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __iter__(self):
        return self 

    def __next__(self):
        self.current_index += 1
        if self.current_index >= self.max_index:
            return np.array([]) 

        return self.dataset[self.current_index]

    def is_next(self):
        return self.current_index < self.max_index 



    
            
