from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from lightning import LightningModule

from datasets.ascad.ascadv1 import ASCADv1_Targets
from utils.aes import AES_INV_SBOX

def get_accuracy(logits: torch.Tensor, labels: torch.Tensor, avg_result: bool = True) -> torch.Tensor:
    rv = (torch.argmax(logits, dim=-1) == labels).to(torch.float)
    if avg_result:
        rv = rv.mean()
    return rv

def get_rank(logits: torch.Tensor, labels: torch.Tensor, avg_result: bool = True) -> torch.Tensor:
    correct_logits = logits.gather(1, labels.unsqueeze(1))
    rank = (logits >= correct_logits).to(torch.float).sum(dim=1)
    if avg_result:
        rank = rank.mean()
    return rank

@torch.no_grad()
def test_side_channel_attacker(module: LightningModule):
    module.eval()
    attack_dataloader = module.test_dataloader()
    attack_dataloader.dataset.target = [ASCADv1_Targets.KEY, ASCADv1_Targets.PLAINTEXT]
    batch_size = attack_dataloader.batch_size
    trace_count = len(attack_dataloader.dataset)
    collected_logits = np.full((trace_count, 16, 256), np.nan, dtype=np.float32)
    collected_keys = np.full((trace_count, 16), -1, dtype=np.int16)
    collected_plaintexts = np.full((trace_count, 16), -1, dtype=np.int16)
    for bidx, batch in enumerate(attack_dataloader):
        trace, metadata = batch
        trace = trace.to(module.device)
        key, plaintext = metadata.split((16, 16), dim=1)
        logits = module(trace)
        logits, key, plaintext = map(lambda x: x.cpu().numpy(), (logits, key, plaintext))
        collected_logits[bidx*batch_size:bidx*batch_size+len(logits), :, :] = logits
        collected_keys[bidx*batch_size:bidx*batch_size+len(key), :] = key
        collected_plaintexts[bidx*batch_size:bidx*batch_size+len(plaintext), :] = plaintext
    assert np.all(np.isfinite(collected_logits))
    assert np.all(collected_keys >= 0)
    assert np.all(collected_plaintexts >= 0)
    collected_keys = collected_keys.astype(np.uint8)
    collected_plaintexts = collected_plaintexts.astype(np.uint8)
    ranks = np.full((trace_count, 16), -1, dtype=np.int16)
    accumulated_predicted_dist = np.ones((16, 256), dtype=np.float32)/256.
    for idx, (logits, key, plaintext) in enumerate(zip(collected_logits, collected_plaintexts, collected_keys)):
        ranks[idx, :] = get_rank(torch.from_numpy(accumulated_predicted_dist), torch.from_numpy(key), avg_result=False).numpy()
        logits_for_key = logits[:, plaintext ^ AES_INV_SBOX[np.arange(256)]]
        dist_over_key = nn.functional.log_softmax(torch.from_numpy(logits_for_key)).numpy()
        accumulated_predicted_dist += dist_over_key
    return ranks