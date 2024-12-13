def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break


import sys
for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        pass
    elif _backend == "pytorch":
        from .freedom.FREEDOM import FREEDOM
        from .bprmf.BPRMF import BPRMF
        from .vbpr.VBPR import VBPR
        from .lightgcn.LightGCN import LightGCN
        from .lightgcn_m.LightGCNM import LightGCNM
        from .sgl.SGL import SGL
        from .ngcf.NGCF import NGCF
        from .ngcf_m.NGCFM import NGCFM
        from .bm3.BM3 import BM3
        from .mgcn.MGCN import MGCN
        from .lgmrec.LGMRec import LGMRec
