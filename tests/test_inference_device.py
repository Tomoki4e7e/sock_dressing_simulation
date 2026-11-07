import os

import pytest

from sock_dressing_simulation.inference_device import device_decision


def test_auto_uses_idle_gpu_with_enough_memory():
    result = device_decision('auto', True, 6991, [dict(pid=-1, memory_mib=0)])
    assert result['device'] == 'cuda'


@pytest.mark.parametrize('available,free,processes,error', [
    (False, 0, [], None),
    (True, 1024, [], None),
    (True, 6991, [dict(pid=-1, memory_mib=1024)], None),
    (True, 6991, [], 'workload query failed'),
])
def test_auto_avoids_unavailable_busy_or_unverified_gpu(available, free, processes, error):
    result = device_decision('auto', available, free, processes, inspection_error=error)
    assert result['device'] == 'cpu' and result['reason']


def test_own_gpu_allocations_do_not_count_as_competition():
    assert device_decision('auto', True, 5000, [dict(pid=os.getpid(), memory_mib=1024)])['device'] == 'cuda'


def test_explicit_cuda_does_not_silently_fall_back_when_busy():
    with pytest.raises(RuntimeError, match='Another GPU compute process'):
        device_decision('cuda', True, 5000, [dict(pid=-1, memory_mib=1000)])


def test_explicit_cpu_is_respected_even_when_gpu_is_available():
    assert device_decision('cpu', True, 6991)['device'] == 'cpu'
