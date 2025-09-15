import torch


def send_batch_to_device(batch, device, return_batch=False):
    """
    Sends all the PyTorch tensors in the batch (a dictionary)
    to the specified device.
    """
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device=device)

    if return_batch:
        return batch
