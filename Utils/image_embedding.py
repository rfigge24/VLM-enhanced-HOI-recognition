"""This module contains all the scripts for preprocessing the images using the pretrained CLIP model.

The functions will preproces the images, 
and get to the CLIP embeddings using the CLIPProcessor and CLIPModel.

"""


import os
import math
import torch
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_DEVICE)


def pad_image_to_square(image, padding_color) -> Image.Image:
    """Pads the shortest side so this image becomes a square.

    Args:
        image: The image to pad.
        padding_color: The color of the padding pixels.

    Returns:
        An square image with padding if needed.
    """
    width, height = image.size
    size = max(width,height)
    return ImageOps.pad(image, (size, size), color=padding_color)

def images_to_clip_embeddings(input_paths: list[str], 
                         batch_size: int, 
                         padding: bool = False, 
                         out_path: str = None
                         ) -> torch.tensor:
    """Reads images from their paths. 
    To then preproces and encode them using the CLIP model wtih the default pretrained settings. 
    Images their shortest side will be rescaled to 224 px and if the longer side is greated then 224px it will be cropped 
    (Note that the padding option will make sure both sides are the same before processing)

    Args:
        input_paths: A list of path strings of the images.
        batch_size: The batch size for loading and processing the images.
        padding: A boolean indicating if the images need to be padded. 
            If False the images will be center cropped. Defaults to False.
        output_path: The path + filename to save the embeddings in .pt-format.
            If None the embeddings will be returned instead of saved. Defaults to None.

    Returns:
        If output_path is None, a torch tensor containing the embeddings.
    """

    n_batches = math.ceil(len(input_paths) / batch_size)
    embedding_list = []
    
    for i in range(n_batches):
        batch_paths = input_paths[i*batch_size : (i+1)*batch_size]
        batch_images = [Image.open(image_path).convert("RGB") for image_path in batch_paths]

        if padding:
            batch_images = [pad_image_to_square(image, "white") for image in batch_images]

        batch_pixel_values = _PROCESSOR(images=batch_images, return_tensors="pt")["pixel_values"].to(_DEVICE)

        for image in batch_images:
            image.close()

        with torch.no_grad():
            batch_embeddings = _MODEL.get_image_features(batch_pixel_values)
        embedding_list.append(batch_embeddings.to("cpu"))

        torch.cuda.empty_cache()

    embeddings = torch.cat(embedding_list, dim = 0)

    if out_path == None:
        return embeddings

    torch.save(embeddings, out_path+'.pt')


if __name__ == '__main__':
    #Train
    paths1 = ["../hico_20150920/images/train2015/" + fname for fname in os.listdir("../hico_20150920/images/train2015")]
    images_to_clip_embeddings(paths1, 256, padding=False, out_path='Embeddings/hico_train_center_crop')
    images_to_clip_embeddings(paths1, 256, padding=True, out_path='Embeddings/hico_train_square_white_padded')

    #Test
    paths2 = ["../hico_20150920/images/test2015/" + fname for fname in os.listdir("../hico_20150920/images/test2015")]
    images_to_clip_embeddings(paths2, 256, padding=False, out_path='Embeddings/hico_test_center_crop')
    images_to_clip_embeddings(paths2, 256, padding=True, out_path='Embeddings/hico_test_square_white_padded')







