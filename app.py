"""
Per fer test he creat un virtual enviroment amb 'python -m venv env' (a la carpeta)

Per activar-lo: .\env\Scripts\activate
Desactivar: deactivate

Un cop en un virtual enviroment instalar:

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python matplotlib
pip install 'git+https://github.com/facebookresearch/segment-anything.git'

Tambe fa falta descarregar el model https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
Hem de posar el fitxer dins de la carpeta on tenim el codi i el nom del fitxer en la variable sam_checkpoint


"""

import torch
import torchvision
#print("PyTorch version:", torch.__version__)
#print("Torchvision version:", torchvision.__version__)
#print("CUDA is available:", torch.cuda.is_available())
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def process_image(image_path):
            
    image = cv2.imread(image_path)  #Imatge per segmentar
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = "sam_vit_b_01ec64.pth" # Model de dades mes petit
    model_type = "vit_b"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)


    mask_generator_ = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.96,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    with torch.no_grad():  # Disable gradient computation to save memory
        masks = mask_generator_.generate(image)

    # After inference: Clear unused memory
    torch.cuda.empty_cache()

    def show_anns(anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:,:,i] = color_mask[i]
            ax.imshow(np.dstack((img, m*0.35)))


    for mask in masks:
        mask_img = mask['segmentation']
        random_color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)  # Generate a random color
        image[mask_img > 0] = random_color 


    processed_image_dir = './static/processed_images'
    os.makedirs(processed_image_dir, exist_ok=True)  # Ensure the directory exists

    # Construct the full processed image path
    processed_image_path = os.path.join(processed_image_dir, os.path.basename(image_path).replace(".jpg", "_processed.jpg"))

    # Save the image
    cv2.imwrite(processed_image_path, image)

    # Return the relative path for Flask to access the image
    return processed_image_path.replace('./static', '')

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show() 
