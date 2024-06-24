"""This module contains our custom oversampler that makes that each class atleast occures a certain number of times in the training set.

"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random

class OverSampler(Sampler):
    def __init__(
        self, labels: torch.Tensor, min_samples_per_class = 40, shuffle: bool = True
    ) -> None:
        self.labels = labels
        self.min_samples_per_class = min_samples_per_class
        self.shuffle = shuffle
    
    def __iter__(self):
        # Getting the indices of the samples per class
        class_indices = [torch.where(self.labels[:,i] == 1)[0] for i in range(self.labels.shape[1])]

        # Oversample indices of classes with less than min_samples_per_class samples:
        oversampled_indices = []
        for cls, indices in enumerate(class_indices):
            if len(indices) < self.min_samples_per_class:
                oversampled_indices.append(
                    indices[torch.randint(len(indices),(self.min_samples_per_class - len(indices),), dtype=torch.int64)]
                )
            
        new_indices = list(range(len(self.labels)))+torch.cat(oversampled_indices).tolist()
        if self.shuffle:
            random.shuffle(new_indices)
        return iter(new_indices)
    
    def __len__(self):
        original_len = len(self.labels)

        number_of_samples_to_add = torch.sum(self.min_samples_per_class - torch.sum(self.labels == 1, dim = 0)[torch.sum(self.labels == 1, dim = 0) < 40])

        return (original_len + number_of_samples_to_add).item()

    

 