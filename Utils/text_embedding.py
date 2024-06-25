"""This module contains all the scripts for preprocessing the text using the pretrained CLIP model.

The functions will preproces the text, 
and get to the CLIP embeddings using the CLIPProcessor and CLIPModel.

"""

import json
import os
import math
import torch
from transformers import CLIPProcessor, CLIPModel


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_DEVICE)

def text_to_clip_embeddings(text_input: list[list[str]],
                         remove_path_from_list: bool = True,
                         out_path: str = None
                         ) -> torch.tensor:
    """Takes the list containing the lists per sample with text answers as input 
    To then preproces and encode them using the CLIP model with the default pretrained settings.

    Args:
        text_input: A list with for each sample a list of strings containing the answers on the vlm prompts.
        remove_path_from_list: Indicates if the first item of each list of strings is a path that needs to be removed. Defaults to True.
        output_path: The path + filename to save the embeddings in .pt-format.
            If None the embeddings will be returned instead of saved. Defaults to None.

    Returns:
        If output_path is None, a torch tensor containing the embeddings.
    """

    embedding_list = []
    for texts in text_input:
        if remove_path_from_list:
            texts = texts[1:]

        input_ids = _PROCESSOR(texts, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(_DEVICE)

        with torch.no_grad():
            text_embeddings = _MODEL.get_text_features(input_ids)
        embedding_list.append(text_embeddings.to("cpu"))

        torch.cuda.empty_cache()

    embeddings = torch.stack(embedding_list)

    if out_path == None:
        return embeddings

    os.makedirs(os.path.dirname(out_path+'.pt'), exist_ok=True)
    torch.save(embeddings, out_path+'.pt')


if __name__ == '__main__':
    #Train
    with open('VLM_Answers/Processed_Answers/vlm_answers_train_processed.json') as f:
        train_vlm_text_answers = json.load(f)

    #   Removing the first Answer on the Questions since it is not HOI related:
    train_vlm_text_answers = [[answer for i,answer in enumerate(answerlist) if i != 1] for answerlist in train_vlm_text_answers]

    out_path_train = 'Embeddings/Text_Embeddings/train'
    text_to_clip_embeddings(train_vlm_text_answers, out_path=out_path_train, remove_path_from_list= True)

    #Test
    with open('VLM_Answers/Processed_Answers/vlm_answers_test_processed.json') as f:
        test_vlm_text_answers = json.load(f)
        
     #   Removing the first Answer on the Questions since it is not HOI related:
    test_vlm_text_answers = [[answer for i,answer in enumerate(answerlist) if i != 1] for answerlist in test_vlm_text_answers]
    
    out_path_test = 'Embeddings/Text_Embeddings/test'
    text_to_clip_embeddings(test_vlm_text_answers, out_path=out_path_test, remove_path_from_list= True)
