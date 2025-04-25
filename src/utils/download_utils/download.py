import os
from math import ceil

import requests
from tqdm import tqdm

def download(url, dest, force=False, chunk_size=2**20, verbose=False):
    dest_dir, dest_name = os.path.split(dest)
    if force or not(os.path.exists(dest)):
        if os.path.exists(dest):
            os.remove(dest)
        os.makedirs(dest_dir, exist_ok=True)
        if verbose:
            print(f'Downloading to path \'{dest}\' from url \'{url}\'...')
        response = requests.get(url, stream=True)
        with open(dest, 'wb') as f:
            if verbose:
                data_iter = tqdm(
                    response.iter_content(chunk_size=chunk_size),
                    total=ceil(int(response.headers['Content-length'])/chunk_size),
                    unit='MB'
                )
            else:
                data_iter = response.iter_content(chunk_size=chunk_size)
            for data in data_iter:
                f.write(data)