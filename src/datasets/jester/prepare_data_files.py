from typing import Literal, Dict, Tuple, Union
import os
import subprocess
import tarfile
import io

from tqdm import tqdm
import numpy as np

from utils.download_utils import download, verify_sha256

_URLS = [
    r'https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Jester/20bnjester-v1-00',
    r'https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Jester/20bnjester-v1-01',
    r'https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Jester/20bnjester-v1-02'
]
_SHA_HASHES = [
    r'477d047b06dca56a3ceb09026840afc5aa1a3f3318e4901dc772e51d43a1d8a7',
    r'93b840e531c4d785d33f9a334b8448c21d90895dad144bdf27555c1e2cec2f31',
    r'03bfb430c0be0c70517de0cd0efb09a87e6e7c6e4c4d5d7600ead7c76ef983e8'
]
_FILENAMES = [url.split('/')[-1] for url in _URLS]
_ANNOTATION_URLS = [
    r'https://github.com/udacity/CVND---Gesture-Recognition/raw/refs/heads/master/20bn-jester-v1/annotations/jester-v1-labels-quick-testing.csv',
    r'https://github.com/udacity/CVND---Gesture-Recognition/raw/refs/heads/master/20bn-jester-v1/annotations/jester-v1-labels.csv',
    r'https://github.com/udacity/CVND---Gesture-Recognition/raw/refs/heads/master/20bn-jester-v1/annotations/jester-v1-test.csv',
    r'https://github.com/udacity/CVND---Gesture-Recognition/raw/refs/heads/master/20bn-jester-v1/annotations/jester-v1-train-quick-testing.csv',
    r'https://github.com/udacity/CVND---Gesture-Recognition/raw/refs/heads/master/20bn-jester-v1/annotations/jester-v1-train.csv',
    r'https://github.com/udacity/CVND---Gesture-Recognition/raw/refs/heads/master/20bn-jester-v1/annotations/jester-v1-validation-quick-testing.csv',
    r'https://github.com/udacity/CVND---Gesture-Recognition/raw/refs/heads/master/20bn-jester-v1/annotations/jester-v1-validation.csv'
]
_ANNOTATION_FILENAMES = [url.split('/')[-1] for url in _ANNOTATION_URLS]

def download_data_files(root: str):
    for url, sha_hash, filename in zip(_URLS, _SHA_HASHES, _FILENAMES):
        filepath = os.path.join(root, filename)
        if os.path.exists(filepath) and not(verify_sha256(filepath, sha_hash)):
            os.remove(filepath)
        if not os.path.exists(filepath):
            download(url, filepath, verbose=True)
        assert os.path.exists(filepath) and verify_sha256(filepath, sha_hash)
    for url, filename in zip(_ANNOTATION_URLS, _ANNOTATION_FILENAMES):
        filepath = os.path.join(root, filename)
        if not os.path.exists(filepath):
            download(url, filepath, verbose=False)

def extract_data_files(root: str):
    cat_path = os.path.join(root, '20bn-jester-v1-cat')
    if not os.path.exists(cat_path):
        with open(cat_path, 'wb') as cat_f:
            for filename, sha_hash in zip(_FILENAMES, _SHA_HASHES):
                filepath = os.path.join(root, filename)
                size = os.path.getsize(filepath)
                with open(filepath, 'rb') as part_f:
                    for chunk in tqdm(iter(lambda: part_f.read(1024**2), b''), total=size//(1024**2), desc=f'Writing {filename}', unit='MB'):
                        cat_f.write(chunk)
    if not os.path.exists(os.path.join(root, 'done_extracting')):
        with tarfile.open(cat_path, mode='r:gz') as tar:
            for member in tqdm(tar, desc=f'Extracting {cat_path.split(os.sep)[-1]}', unit='files'):
                tar.extract(member, path=root)
        with open(os.path.join(root, 'done_extracting'), 'w') as _: pass

def load_classes(root: str) -> Dict[str, int]:
    label_filename = r'jester-v1-labels.csv'
    label_filepath = os.path.join(root, label_filename)
    assert os.path.exists(label_filepath)
    with open(label_filepath, 'r') as f:
        classes = {line.strip(): idx for idx, line in enumerate(f)}
    return classes

def load_labels_and_indices(root: str, split: Literal['train', 'validation', 'test']) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    assert split in ['train', 'validation', 'test']
    classes = load_classes(root)
    filename = f'jester-v1-{split}.csv'
    filepath = os.path.join(root, filename)
    assert os.path.exists(filepath)
    data_indices = []
    if split != 'test':
        data_labels = []
    with open(filepath, 'r') as f:
        for line in f:
            if split == 'test':
                data_idx = line.strip()
                data_indices.append(int(data_idx))
            else:
                data_idx, data_label = line.strip().split(';')
                data_indices.append(int(data_idx))
                data_labels.append(classes[data_label])
    data_indices = np.array(data_indices)
    if split != 'test':
        data_labels = np.array(data_labels)
        return data_indices, data_labels
    else:
        return data_indices

def convert_to_webd_format(root: str, shard_size: int = 1000):
    for split in ['train', 'validation', 'test']:
        shards_dir = os.path.join(root, f'{split}_shards')
        os.makedirs(shards_dir, exist_ok=True)
        if split == 'test':
            data_indices = load_labels_and_indices(root, split)
        else:
            data_indices, data_labels = load_labels_and_indices(root, split)
        tar = None
        shard_idx = 0
        rng = np.random.default_rng(seed=0)
        perm_indices = rng.permutation(len(data_indices))
        # can't fully shuffle with streaming dataloader, so I want to ensure data is already mixed beforehand
        data_indices = data_indices[perm_indices]
        if split != 'test':
            data_labels = data_labels[perm_indices]
        for idx, data_idx in tqdm(enumerate(data_indices)):
            key = f'{idx}'
            if idx % shard_size == 0:
                if tar is not None:
                    tar.close()
                shard_path = os.path.join(shards_dir, f'shard-{shard_idx:06d}.tar')
                should_archive = not os.path.exists(shard_path)
                if should_archive:
                    tar = tarfile.open(shard_path, 'w')
                shard_idx += 1
            if should_archive:
                video_dir = os.path.join(root, '20bn-jester-v1', f'{data_idx}')
                frame_indices = sorted([int(x.split('.')[0]) for x in os.listdir(video_dir) if x.endswith('.jpg')])
                for frame_idx in frame_indices:
                    frame_path = os.path.join(video_dir, f'{frame_idx:05}.jpg')
                    with open(frame_path, 'rb') as f:
                        frame_bytes = f.read()
                    tarinfo = tarfile.TarInfo(name=f'{key}.{frame_idx:03d}.jpg')
                    tarinfo.size = len(frame_bytes)
                    tar.addfile(tarinfo, io.BytesIO(frame_bytes))
                if split != 'test':
                    label = data_labels[idx]
                    label_bytes = str(label).encode('utf-8')
                    label_info = tarfile.TarInfo(name=f'{key}.cls')
                    label_info.size = len(label_bytes)
                    tar.addfile(label_info, io.BytesIO(label_bytes))
        if tar is not None:
            tar.close()