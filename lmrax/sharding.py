import flax.traverse_util as traverse_util
import jax.sharding as shd
import numpy as np


def get_params_shardings(mesh, pytree):
    def _to_shardings(k, v):
        if not isinstance(v, np.ndarray) or v.ndim == 0:
            return shd.NamedSharding(mesh, shd.PartitionSpec())
        if "Attention" in k:
            if "relative_attention" in k:
                _spec_tuple = (None, None)
            elif "Attention.o" in k:  # o project
                _spec_tuple = ("tp", None)
            else:
                _spec_tuple = (None, "tp")  # q, k, v projects
        elif "DenseReluDense" in k:
            if "DenseReluDense.wo" in k:
                _spec_tuple = ("tp", None)
            else:
                _spec_tuple = (None, "tp")
        elif "shared" in k:
            _spec_tuple = ("tp", None)
        else:
            assert v.ndim == 1
            _spec_tuple = ("tp",)
        assert len(_spec_tuple) == v.ndim
        # _spec_tuple = reversed(_spec_tuple)
        # _spec_tuple = [None] * len(_spec_tuple)
        # _spec_tuple[-1] = "tp"
        p = shd.PartitionSpec(*_spec_tuple)
        return shd.NamedSharding(mesh, p)

    flat = traverse_util.flatten_dict(pytree, sep=".")
    for k, v in flat.items():
        flat[k] = _to_shardings(k, v)
    return traverse_util.unflatten_dict(flat, sep=".")


def get_batch_shardings(mesh, inputs):
    return_dict = {}
    for k, v in inputs.items():
        if not isinstance(v, np.ndarray) or v.ndim == 0:
            _spec_tuple = []
        elif "input_ids" in k or "labels" in k or "mask" in k:
            _spec_tuple = [None] * (v.ndim)
            _spec_tuple[0] = "dp"
        else:
            _spec_tuple = []
        p = shd.PartitionSpec(*_spec_tuple)
        return_dict[k] = shd.NamedSharding(mesh, p)
    return return_dict
