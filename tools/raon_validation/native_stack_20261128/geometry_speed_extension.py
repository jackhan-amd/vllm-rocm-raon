"""Process-local native geometry candidate; no numerical or packing changes."""
from functools import cache
import json
import os
from pathlib import Path
import raon_compile_warmup
from vllm.models.raon1_mini import rocm_dense
from aiter.ops.flydsl.kernels.gemm_a16w16_skinny_gfx950 import skinny_config_error

class GeometryExtension:
    pass

policy=os.environ['RAON_GEOMETRY_POLICY'];assert policy in ('baseline','dense','both')
original=rocm_dense.configuration
records=[]
@cache
def configured(rows,n,k):
    base=original(rows,n,k)
    if base is None or policy=='baseline':return base
    tile=None
    if (n,k)==(2176,2560):tile=16
    elif (n,k)==(1152,2560):tile=8
    elif (n,k)==(5120,2560) and rows<=8:tile=8 if rows==1 else 32
    elif (n,k)==(24064,2560) and rows<=8:tile=16 if rows==1 else 32
    elif (n,k)==(2560,51200):tile=16
    if tile is None:return base
    reason=skinny_config_error(n,k,base['block_m'],tile,base['block_k'],base['stages'],base['k_waves'])
    assert reason is None,('Unsupported geometry',rows,n,k,tile,reason)
    candidate=dict(base,block_n=tile)
    records.append(dict(rows=rows,n=n,k=k,baseline=base,candidate=candidate))
    path=Path(os.environ['RAON_CASE_OUT'])/f'geometry-policy-{os.getpid()}.json'
    path.write_text(json.dumps(dict(policy=policy,records=records,precision_changed=False,packing_changed=False),indent=2))
    return candidate
rocm_dense.configuration=configured
