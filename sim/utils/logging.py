from dataclasses import fields, is_dataclass
import numpy as np


def flatten_state(s, prefix="", skip=("debug",)):
    out = {}
    for f in fields(s):
        if f.name in skip:
            continue
        val = getattr(s, f.name)
        key = f"{prefix}{f.name}"
        if is_dataclass(val):
            out.update(flatten_state(val, prefix=f"{key}."))
        else:
            out[key] = val
    return out


def log_to_arrays(log_list):
    flat = [flatten_state(s) for s in log_list]
    out = {}
    for k in flat[0]:
        try:
            out[k] = np.asarray([d[k] for d in flat])
            if out[k].dtype == object:
                print(f"WARNING: {k} produced object array")
        except ValueError:
            shapes = {np.shape(d[k]) for d in flat}
            raise ValueError(f"Field '{k}' has inconsistent shapes: {shapes}")
    return out