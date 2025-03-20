import os
import time
from multiprocessing import current_process

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import get_worker_info

def get_trace_sample_stats(_traces, cache_path, indices=None, chunk_size=10000):
    if indices is None:
        indices = np.arange(len(_traces))
    worker_info = get_worker_info()
    if worker_info is None:
        worker_id = -1
    else:
        worker_id = worker_info.id
    if not(os.path.exists(cache_path)) and worker_id <= 0:
        traces = _traces
        if isinstance(traces, torch.Tensor):
            traces = traces.numpy()
        trace_count, trace_dim = traces.shape
        mean = np.zeros((trace_dim,), dtype=np.float32)
        var = np.zeros((trace_dim,), dtype=np.float32)
        count = 0
        progress_bar = tqdm(total=2*len(indices))
        for chunk_idx in range(trace_count//chunk_size):
            traces_chunk = np.array(traces[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size, :]).astype(np.float32)
            mask = (indices >= chunk_idx * chunk_size) & (indices < (chunk_idx + 1) * chunk_size)
            selected_indices = indices[mask] - (chunk_idx * chunk_size)
            traces_chunk = traces_chunk[selected_indices, :]
            if len(traces_chunk) > 0:
                mean = (count/(count+len(traces_chunk)))*mean + (chunk_size/(count+len(traces_chunk)))*traces_chunk.mean(axis=0)
            count += len(traces_chunk)
            progress_bar.update(len(traces_chunk))
        for chunk_idx in range(trace_count//chunk_size):
            traces_chunk = np.array(traces[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size, :]).astype(np.float32)
            mask = (indices >= chunk_idx * chunk_size) & (indices < (chunk_idx + 1) * chunk_size)
            selected_indices = indices[mask] - (chunk_idx * chunk_size)
            traces_chunk = traces_chunk[selected_indices, :]
            if len(traces_chunk) > 0:
                var = (count/(count+len(traces_chunk)))*var + (chunk_size/(count+len(traces_chunk)))*((traces_chunk-mean)**2).mean(axis=0)
            count += len(traces_chunk)
            progress_bar.update(len(traces_chunk))
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