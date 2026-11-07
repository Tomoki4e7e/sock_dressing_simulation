import sys
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from process_lit_reaching import mask_quality


def test_partial_shadow_mask_is_rejected_even_with_perfect_mesh_precision():
    sock=np.zeros((64,64),bool);sock[20:60,20:50]=True
    foot=np.zeros_like(sock);foot[0:18,20:50]=True
    partial=sock.copy();partial[40:]=False
    checks,areas,precision,iou=mask_quality(dict(sock=partial,foot=foot),dict(sock=sock,foot=foot),sock)
    assert precision['sock']==1.0 and iou==.5
    assert not checks['visible_sock_iou']
    checks,_,_,iou=mask_quality(dict(sock=sock,foot=foot),dict(sock=sock,foot=foot),sock)
    assert iou==1.0 and all(checks.values())
