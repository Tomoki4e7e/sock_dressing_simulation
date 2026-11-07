"""Select CUDA only when visible, sufficiently free, and not busy computing."""
from __future__ import annotations

import os
import subprocess


def device_decision(request, cuda_available, free_mib=0, compute_processes=(),
                    minimum_free_mib=3072, inspection_error=None):
    if request not in ('auto', 'cpu', 'cuda'):
        raise ValueError(f'Unknown inference device: {request}')
    reason = None
    if request == 'cpu':reason = 'CPU explicitly requested'
    elif not cuda_available:reason = 'CUDA is unavailable in this execution environment'
    elif inspection_error:reason = f'GPU workload inspection failed: {inspection_error}'
    elif free_mib < minimum_free_mib:reason = 'Insufficient free GPU memory'
    elif any(p['memory_mib'] > 256 and p['pid'] != os.getpid() for p in compute_processes):
        reason = 'Another GPU compute process is active'
    if request == 'cuda' and reason:
        raise RuntimeError(f'Cannot use CUDA for inference: {reason}')
    return dict(requested=request, device='cpu' if reason else 'cuda',
                reason=reason or 'CUDA is available with sufficient free memory and no competing compute workload',
                free_memory_mib=free_mib, minimum_free_memory_mib=minimum_free_mib,
                compute_processes=list(compute_processes))


def select_inference_device(request='auto', minimum_free_mib=3072):
    import torch
    available = request != 'cpu' and torch.cuda.is_available()
    free = 0;processes = [];error = None
    if available:
        try:
            free = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True, timeout=10).strip())
            output = subprocess.check_output(['nvidia-smi', '--id=0',
                '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True, timeout=10)
            for line in output.splitlines():
                if not line.strip():continue
                pid, memory = line.split(',')
                # Unknown usage cannot establish that the process is idle.
                used = int(memory.strip()) if memory.strip().isdigit() else 257
                processes.append(dict(pid=int(pid.strip()), memory_mib=used))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:error = str(exc)
    report = device_decision(request, available, free, processes, minimum_free_mib, error)
    device = torch.device(report['device'])
    if device.type == 'cuda':report['gpu_name'] = torch.cuda.get_device_name(0)
    return device, report
