import os
import time
from multiprocessing import current_process

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import get_worker_info

def get_trace_sample_stats(_traces, cache_path, indices=None, chunk_size=100):
    if indices is None:
        indices = np.arange(len(traces))
    worker_info = get_worker_info()
    if worker_info is None:
        worker_id = -1
    else:
        worker_id = worker_info.id
    if not(os.path.exists(cache_path)) and worker_id <= 0:
        traces = _traces
        if isinstance(traces, torch.Tensor):
            traces = traces.numpy()
        trace_count = len(indices)
        _, trace_dim = traces.shape
        mean = np.zeros((trace_dim,), dtype=np.float32)
        var = np.zeros((trace_dim,), dtype=np.float32)
        count = 0
        progress_bar = tqdm(total=2*trace_count)
        for chunk_idx in range(trace_count//chunk_size):
            traces_chunk = np.array(traces[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size, :]).astype(np.float32)
            mean = (count/(count+chunk_size))*mean + (chunk_size/(count+chunk_size))*traces_chunk.mean(axis=0)
            count += chunk_size
            progress_bar.update(chunk_size)
        for chunk_idx in range(trace_count//chunk_size):
            traces_chunk = np.array(traces[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size, :]).astype(np.float32)
            var = (count/(count+chunk_size))*var + (chunk_size/(count+chunk_size))*((traces_chunk-mean)**2).mean(axis=0)
            count += chunk_size
            progress_bar.update(chunk_size)
        std = np.sqrt(var)
        np.save(cache_path, np.stack([mean, std]))
    elif worker_id > 0:
        while not os.path.exists(cache_path):
            time.sleep(0.1)
    else:
        print('Stats cache exists already.')
    mean_and_std = np.load(cache_path)
    mean = mean_and_std[0, :]
    std = mean_and_std[1, :]
    return mean, std