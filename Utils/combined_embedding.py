"""This module contains the function to combine the image and text embeddings made previously.

"""

import torch
import os

def combine_image_and_text_embeddings(image_embeddings_path: str,
               text_embeddings_path: str,
               out_path: str = None):
    """This function concatenates the image embedding with the text embeddings.

    Args:
        image_embeddings_path: The path to the image embeddings pt file.
        text_embeddings_path: The path to the text embeddings pt file.
        out_path: The output path + file name to save the combined embedding to. Defaults to None.

    Returns:
        None if an out_path is specified, else a PyTorch tensor containing the combined embeddings.
    """    
    image_embeddings = torch.load(image_embeddings_path).unsqueeze(1)
    text_embeddings = torch.load(text_embeddings_path)

    combined_embeddings = torch.cat((image_embeddings,text_embeddings), dim = 1)

    if out_path == None:
        return combined_embeddings

    os.makedirs(os.path.dirname(out_path+'.pt'), exist_ok=True)
    torch.save(combined_embeddings, out_path+'.pt')


if __name__ == '__main__':
    # Train:
    train_image_embeddings_path = 'Embeddings/hico_train_center_crop.pt'
    train_image_embeddings_path_2 = 'Embeddings/hico_train_square_white_padded.pt'
    train_text_embeddings_path = 'Embeddings/Text_Embeddings/train.pt'
    combine_image_and_text_embeddings(train_image_embeddings_path,train_text_embeddings_path, out_path='Embeddings/Combined_Embeddings/train')
    combine_image_and_text_embeddings(train_image_embeddings_path_2,train_text_embeddings_path, out_path='Embeddings/Combined_Embeddings/train_whitepadded')

    # Test:
    test_image_embeddings_path = 'Embeddings/hico_test_center_crop.pt'
    test_image_embeddings_path_2 = 'Embeddings/hico_test_square_white_padded.pt'
    test_text_embeddings_path = 'Embeddings/Text_Embeddings/test.pt'
    combine_image_and_text_embeddings(test_image_embeddings_path,test_text_embeddings_path, out_path='Embeddings/Combined_Embeddings/test')
    combine_image_and_text_embeddings(test_image_embeddings_path_2,test_text_embeddings_path, out_path='Embeddings/Combined_Embeddings/test_whitepadded')