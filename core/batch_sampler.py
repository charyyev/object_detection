import os
import numpy as np
import random


class BatchSampler():
    def __init__(self, data_file, batch_size, drop_last=False):
        self.data_list = []
        self.batch_size = batch_size
        self.data_list = read_data_file(data_file)
        self.num_samples = len(self.data_list)
        self.drop_last = drop_last
        self.batches = self.get_batches()
        self.num_batches = len(self.batches)

    def get_batches(self):
        perm_indices = random.sample(self.data_list, len(self.data_list))
        last_ind = self.num_samples - self.num_samples % self.batch_size
        batches = np.split(perm_indices[0:last_ind], self.num_samples // self.batch_size)
        
        if not self.drop_last and last_ind != self.num_samples:
            batches.append(perm_indices[last_ind:])

        return batches
    
    
    def __iter__(self):
        yield from self.batches
        
    def __len__(self):
        return self.num_batches


def read_data_file(data_file):
    data_indices = []
    with open(data_file, "r") as f:
        for line in f:
            line = line.strip()
            data_indices.append(line.split(";")[0])

    return data_indices


if __name__ == "__main__":
    data_folder = "/home/stpc/data/kitti/velodyne/training/velodyne"
    data_file = "/home/stpc/data/train/val.txt"

    batch_sampler = BatchSampler(data_file, 5)

    for batch in batch_sampler:
        print(batch)