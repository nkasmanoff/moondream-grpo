import torch
import torch.distributed as dist

def synchronize_metrics(metrics, device):
    """
    Synchronize metrics across all processes.
    """
    if not dist.is_initialized():
        return metrics
        
    metrics_tensor = torch.tensor(metrics, device=device)
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    return metrics_tensor.tolist()
