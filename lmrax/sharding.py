import flax.traverse_util as traverse_util
import jax.sharding as shd
import numpy as np


def _to_t5_shardings(mesh, k, v):
    if not isinstance(v, np.ndarray) or v.ndim == 0:
        # usually scalars
        return shd.NamedSharding(mesh, shd.PartitionSpec())
    if "Attention" in k and ("kernel" in k or "embedding" in k):
        if "relative_attention" in k:
            # relative attention
            _spec_tuple = (None, None)
        elif "Attention.o" in k:
            # o project
            _spec_tuple = ("tp", None)
        else:
            # q, k, v projects
            _spec_tuple = (None, "tp")
    elif "DenseReluDense" in k:
        if "DenseReluDense.wo" in k:
            # FFN o project
            _spec_tuple = ("tp", None)
        else:
            # FFN in projects
            _spec_tuple = (None, "tp")
    elif "shared" in k:
        # shared embeddings
        _spec_tuple = ("tp", None)
    elif "lm_head" in k:
        # lm head embeddings
        _spec_tuple = (None, "tp")
    else:
        # usually layer norm or bias
        assert v.ndim == 1
        _spec_tuple = (None,)
    assert len(_spec_tuple) == v.ndim
    p = shd.PartitionSpec(*_spec_tuple)
    return shd.NamedSharding(mesh, p)


def get_params_shardings(mesh, pytree, model_name="t5"):
    if "t5" in model_name:
        _to_shardings = _to_t5_shardings
    else:
        raise NotImplementedError(
            f"Sharding for {model_name} is not implemented yet. "
            f"Please open an issue. "
            f"You can also implement it and send a PR."
        )

    flat = traverse_util.flatten_dict(pytree, sep=".")
    for k, v in flat.items():
        flat[k] = _to_shardings(mesh, k, v)
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
