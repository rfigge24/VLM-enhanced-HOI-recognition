"""This file contains a train_val_split function for the hico dataset.

"""
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def train_val_split_hico(X: torch.Tensor, y: torch.Tensor, test_size=0.2, random_state = None):
    """A iterative-stratified shuffled splitter. 
    Following the "HICO: A Benchmark for Recognizing Human-Object Interactions in Images" paper we use a 80-20 split with atleast 1 instance per class for the trainingset.

    Args:
        X: A torch.Tensor with the features shape = (instances, features).
        y: A torch.Tensor with the multi-labeled annotations shape = (instances, classes).
        test_size: The fraction of the size of the test-set. Defaults to 0.2.
        random_state: Option to be able to replicate the same split based on a seed. Defaults to None.
    
    Returns:
        A Tuple of (training indices, validation indices).
    """
    # Initialises a iterative-stratification shuffled splitter:
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    # Creating the splits based on only the HOI-classes even if the augmented annotations are used: (thats why :600)
    train_index, test_index = next(msss.split(X, y[:,:600]))
    train_index, test_index = torch.tensor(train_index), torch.tensor(test_index)

    # Making sure the trainset has atleast 1 sample per class:
    no_instance_classes_idx = torch.nonzero(torch.sum(y[train_index, :600] == 1, dim=0)==0).flatten().tolist()
    indices_to_switch = []
    while len(no_instance_classes_idx) > 0:
        test_set = y[test_index,:600]

        # Find the test_set index of the first instance that satisfies the first no_instance_classes class:
        instance_test_set_idx = torch.nonzero(test_set[:,no_instance_classes_idx[0]] == 1)[0,0]
        # Translate this index to the the y indexing:
        instance_y_idx = test_index[instance_test_set_idx]
        # add this index to the indices list:
        indices_to_switch.append(instance_y_idx)
        # Check for what classes this instance satisfies (could be more than the one we looked for):
        satisfied_classes = torch.nonzero(test_set[instance_test_set_idx, no_instance_classes_idx] == 1).flatten()
        # Remove the satisfied classes from the no_instance_classes:
        no_instance_classes_idx = [cls for (i,cls) in enumerate(no_instance_classes_idx) if not i in satisfied_classes]

    indices_to_switch = torch.tensor(indices_to_switch)

    #   Removing the indices from the testset: 
    mask = torch.isin(test_index,indices_to_switch, invert=True)
    test_index= test_index[mask]
    
    #   Adding the indices to the trainset:
    train_index = torch.cat((train_index, indices_to_switch))

    return (train_index, test_index)