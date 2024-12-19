from __init__ import *

import os
import json
import cloudinary
import cloudinary.uploader
import cloudinary.api
from cloudinary.utils import cloudinary_url

from tqdm import tqdm

def create_json_from_cloudinary() -> None:
    
    file = open('data/cloudinary.txt', 'r')
    
    # Configuration       
    cloudinary.config( 
        cloud_name = file.readline().strip(), 
        api_key = file.readline().strip(), 
        api_secret = file.readline().strip(), # Click 'View API Keys' above to copy your API secret
        secure=True
    )

    file.close()
    
    # The folder name in the cloudinary bucket
    folder_name = 'strawml'
    
    # Get the list of images in the cloudinary bucket
    print('Getting the list of images in the cloudinary bucket...')
    response = cloudinary.api.resources(asset_folder=folder_name)
    
    resources = response['resources']
    if 'next_cursor' in response:
        next_cursor = response['next_cursor']
        while next_cursor:
            response = cloudinary.api.resources(asset_folder=folder_name, next_cursor=next_cursor)
            if 'next_cursor' in response:
                next_cursor = response['next_cursor']
            else:
                next_cursor = None
            resources += response['resources']
        
    
    # Create the JSON file
    image_data = []
    print('Creating the JSON file...')
    for resource in resources:
        image_info = {
            # 'public_id': resource['public_id'],
            'url': resource['secure_url'],
            # 'secure_url': resource['secure_url'],
        }
        image_data.append(image_info)
    
    json_file = open('data/processed/cloudinary.json', 'w')
    json.dump(image_data, json_file, indent=4)
    
    print('JSON file created!')

if __name__ == '__main__':
    create_json_from_cloudinary()