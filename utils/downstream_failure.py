"""Structured failures: capacity fallback must not hide a non-capacity error."""
import torch


def failure_kind(error):
    if isinstance(error, torch.cuda.OutOfMemoryError) and 'CUDA' in str(error):
        return 'cuda_oom'
    if isinstance(error, RuntimeError) and any(text in str(error) for text in (
            'Connection closed by peer', 'Connection reset by peer', 'NCCL communicator was aborted')):
        return 'peer_shutdown'
    return 'non_capacity'


def capacity_failure(records):
    kinds = [row['kind'] for row in records]
    return 'cuda_oom' in kinds and set(kinds).issubset({'cuda_oom', 'peer_shutdown'})
