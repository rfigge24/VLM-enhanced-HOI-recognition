"""This module is used to augment the annotations of the HICO dataset. 
Using the function you can create two seperate annotation sets for the nouns and verbs only.

"""

from collections import defaultdict
import pandas as pd
import torch
import json

def _load_csv_to_tensor(csv_path: str):
    """A helper function for internal use to load csv tables into a torch.tensor.

    Args:
        csv_path: path of the csv file to load.

    Returns:
        the csv table in the form of a torch.Tensor
    """    

    df = pd.read_csv(csv_path, header= None)
    numptensor = df.to_numpy()
    tensor = torch.tensor(numptensor, dtype=torch.float32)

    return tensor


def generate_seperate_annotations(hoi_annotation_path: str, nounlist_path: str, verblist_path: str, hoilist_path: str, out_path_nnames = None, out_path_vnames = None):
    """Function to generate the annotations for the Nouns and Verbs only from the HOI annotations.

    Args:
        hoi_annotation_path: The path to the csv file containing the HOI annotations for the instances.
        nounlist_path: The path to the txt file containing the list of nounnames, in the format of the ones on the HICO website.
        verblist_path: The path to the txt file containing the list of verbnames, in the format of the ones on the HICO website.
        hoilist_path: The path to the json file containing the list of hoi annotations, in the format of the HICO annotation.
        out_path_nnames: A path for the function to output the nnames annotations. Defaults to None, if one output path is None the output is returned.
        out_path_vnames: A path for the function to output the vnames annotations. Defaults to None, if one output path is None the output is returned.

        HICO website: https://websites.umich.edu/~ywchao/hico/

    Returns:
        a tuple containing ((nname_labels, nname_annotations), (vname_labels, vname_annotations))
    """    
    # Read the HOI annotation file and convert to a torch tensor:
    data = _load_csv_to_tensor(hoi_annotation_path)
    n_instances = data.shape[1]

    # Read the HOI labels (a list of dictionaries):
    with open(hoilist_path, 'r') as file:
            hoilist = json.load(file)

    # Create a list of the nounname labels:
    nname_labels = []
    with open(nounlist_path, 'r') as file:
        for line in file.readlines()[2:]:
            name = line.split()[1]
            nname_labels.append(name)
    
    # Create a list of the verbname labels:
    vname_labels = []
    with open(verblist_path, 'r') as file:
        for line in file.readlines()[2:]:
            name = line.split()[1]
            vname_labels.append(name)
    
    # Create dictionaries that maps 'nnames' or 'vnames'to the indices of the HOI classes they are in:
    nname2hoi_indices = defaultdict(list)
    vname2hoi_indices = defaultdict(list)
    for i in range(600):
        nname2hoi_indices[hoilist[i]['nname']].append(i)
        vname2hoi_indices[hoilist[i]['vname']].append(i)
    
    nname_annotations = torch.full((len(nname_labels),n_instances), torch.nan)
    vname_annotations = torch.full((len(vname_labels),n_instances), torch.nan)

    print("Preparation done, annotation process started!")

    # Distilling the noun-name labels for each noun-name class:
    for k,nname in enumerate(nname_labels):
        indices_of_nname = nname2hoi_indices[nname]
        labels_of_nname = data[indices_of_nname,:]
        condition_1 = torch.any(labels_of_nname == 0, dim= 0) 
        condition_2 = torch.any(labels_of_nname == -1, dim= 0)
        condition_3 = torch.any(labels_of_nname == 1, dim=0)

        nname_annotations[k, condition_1] = 0
        nname_annotations[k, condition_2] = -1
        nname_annotations[k, condition_3] = 1

    # Distilling the verb-name labels for each verb-name class:
    for l,vname in enumerate(vname_labels):
        indices_of_vname = vname2hoi_indices[vname]
        labels_of_vname = data[indices_of_vname,:]
        condition_1 = torch.any(labels_of_vname == 0, dim= 0) 
        condition_2 = torch.any(labels_of_vname == -1, dim= 0)
        condition_3 = torch.any(labels_of_vname == 1, dim=0)

        vname_annotations[l, condition_1] = 0
        vname_annotations[l, condition_2] = -1
        vname_annotations[l, condition_3] = 1

    # If one of the output paths is missing return the annotation tensors themselves:
    if out_path_nnames is None or out_path_vnames is None:
        return ((nname_labels, nname_annotations), (vname_labels, vname_annotations))

    # Else write the annotations to their specific output file as csv:
    nname_anno_df = pd.DataFrame(nname_annotations.numpy())
    nname_anno_df.to_csv(out_path_nnames, index=False,header=False, na_rep='NaN') 

    vname_anno_df = pd.DataFrame(vname_annotations.numpy())
    vname_anno_df.to_csv(out_path_vnames, index=False,header=False, na_rep='NaN') 


def augment_HICO_annotations(hoi_annotations: torch.Tensor | str, nname_annotations: torch.Tensor | str, vname_annotations: torch.Tensor | str, out_path = None):
    """Function to combine the annotations into a single table (hoi+nname+vname). That can either be outputted or saved as a csv file

    Args:
        hoi_annotations: The HOI annotations themselves in the form of a tensor or a filepath to the csv file to load from.
        nname_annotations: The nounname annotations themselves in the form of a tensor or a filepath to the csv file to load from.
        vname_annotations: The verbname annotations themselves in the form of a tensor or a filepath to the csv file to load from.
        out_path: An output filepath to save the annotations as a csv. Defaults to None, if None the annotations will be returned.

    Returns:
        If no output file path is specified the augmented annotations as a torch.Tensor.
    """    
    # Load the csv files if the input are path strings else use the input tensors directly:
    hoi_data = _load_csv_to_tensor(hoi_annotations)     if isinstance(hoi_annotations, str) else hoi_annotations
    nname_data = _load_csv_to_tensor(nname_annotations) if isinstance(nname_annotations, str) else nname_annotations
    vname_data = _load_csv_to_tensor(vname_annotations) if isinstance(vname_annotations, str) else vname_annotations

    # Concat the annotations:
    augmented_data = torch.cat((hoi_data, nname_data, vname_data), dim=0)

    # If no output file path is specified return the augmented_data:
    if out_path is None:
        return augmented_data
    
    # Else write the annotations to the output file as csv.
    augmented_data_df = pd.DataFrame(augmented_data.numpy())
    augmented_data_df.to_csv(out_path, index= False, header= False, na_rep='NaN')


def augment_HICO_labels(hoi_labels_input: list[str] | str, nname_labels_input: list[str] | str, vname_labels_input: list[str] | str, out_path = None):
    """Function to combine the seperate labels into a single label list (hoi+nname+vname) for the augmented HICO annotations.

    Args:
        hoi_labels_input: The HOI labels themselves in the form of a list or a filepath to the txt file to load from.
        nname_labels_input: The nname labels themselves in the form of a list or a filepath to the nounlist txt file to load from.
        vname_labels_input: The vname labels themselves in the form of a list or a filepath to the verblist txt file to load from.
        out_path: An output filepath to save the labels as a txt file. Defaults to None, if None the labels will be returned.

    Returns:
        If no output file path is specified the augmented annotations as a list.
    """    
    # Read all the files that are inputted as path strings:
    if isinstance(hoi_labels_input, str):
        hoi_labels =    []
        with open(hoi_labels_input, 'r') as file:
            for line in file.readlines()[2:]:
                words = line.split()
                name = words[1]+'_'+words[2]    # (The HOI labels have two parts)
                hoi_labels.append(name)
    else: hoi_labels = hoi_labels_input

    if isinstance(nname_labels_input, str):
        nname_labels =  []
        with open(nname_labels_input, 'r') as file:
            for line in file.readlines()[2:]:
                name = line.split()[1]
                nname_labels.append(name)
    else: nname_labels = nname_labels_input

    if isinstance(nname_labels_input, str):
        vname_labels =  []
        with open(vname_labels_input, 'r') as file:
            for line in file.readlines()[2:]:
                name = line.split()[1]
                vname_labels.append(name)
    else: vname_labels = vname_labels_input

    # Concat the labels:
    augmented_labels = hoi_labels + nname_labels + vname_labels

    # If no output file path is specified return the augmented_data:
    if out_path is None:
        return augmented_labels

    with open(out_path, 'w') as file:
        for string in augmented_labels:
            file.write(string + '\n')


if __name__ == '__main__':
    train_hoi_annotation_path = "anno/anno_train.csv"
    test_hoi_annotation_path = "anno/anno_test.csv"
    nounlist_path = "anno/added/hico_list_obj.txt"
    verblist_path = "anno/added/hico_list_vb.txt"
    hoilist_path = "anno/added/list_action.json"

    # Generating and saving the seperate annotations for the train and test set:
    generate_seperate_annotations(train_hoi_annotation_path, nounlist_path, verblist_path, hoilist_path, "anno/added/anno_nnames_train.csv", "anno/added/anno_vnames_train.csv")
    generate_seperate_annotations(test_hoi_annotation_path, nounlist_path, verblist_path, hoilist_path, "anno/added/anno_nnames_test.csv", "anno/added/anno_vnames_test.csv")

    # Create the combined annotations csv (rows are in order hoi, nname, vname):
    augment_HICO_annotations(train_hoi_annotation_path, "anno/added/anno_nnames_train.csv", "anno/added/anno_vnames_train.csv", out_path="anno/added/anno_augmented_train.csv" )
    augment_HICO_annotations(test_hoi_annotation_path, "anno/added/anno_nnames_test.csv", "anno/added/anno_vnames_test.csv", out_path="anno/added/anno_augmented_test.csv" )

    # create the combined label_name list for the augmented annotations:
    augment_HICO_labels("anno/added/hico_list_hoi.txt", nounlist_path,verblist_path, out_path= "anno/added/hico_list_augmented.txt")