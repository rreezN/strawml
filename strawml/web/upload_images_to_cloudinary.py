from __init__ import *

import os
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url

from tqdm import tqdm

def upload_to_cloudinary(cloudinary_txt: str, image_folder: str, tag: str = '') -> None:
    
    file = open(cloudinary_txt, 'r')
    
    # Configuration       
    cloudinary.config( 
        cloud_name = file.readline().strip(), 
        api_key = file.readline().strip(), 
        api_secret = file.readline().strip(), # Click 'View API Keys' above to copy your API secret
        secure=True
    )

    file.close()
    
    images_to_upload = os.listdir(image_folder)
    tqdm_images = tqdm(images_to_upload, desc='Uploading images')
    print(f"\nUploading {len(images_to_upload)} images...")
    for i in range(len(images_to_upload)):
        tqdm_images.update()
        id = '.'.join(images_to_upload[i].split('.')[:-1])
        if tag != '':
            upload_result = cloudinary.uploader.upload(f'{image_folder}/{images_to_upload[i]}', asset_folder='strawml', public_id=id, overwrite=True, resource_type='image', tags=tag)
        else:
            upload_result = cloudinary.uploader.upload(f'{image_folder}/{images_to_upload[i]}', asset_folder='strawml', public_id=id, overwrite=True, resource_type='image')
        tqdm_images.set_postfix({'url': upload_result["secure_url"]})
        
        if i > 5:
            break
    
    tqdm_images.close()
    print('Upload complete!')

if __name__ == '__main__':
    upload_to_cloudinary('data/cloudinary.txt', 'data/processed/recordings', tag=['test_tag', 'strawml'])