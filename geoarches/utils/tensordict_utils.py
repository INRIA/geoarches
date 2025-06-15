import torch
from tensordict.tensordict import TensorDict


def tensordict_apply(f, *args, **kwargs):
    tdicts = [a for a in args if isinstance(a, TensorDict)]
    tdicts += [v for v in kwargs.values() if isinstance(v, TensorDict)]
    # check that all found tdicts have same keys
    tdict_keys = set(tdicts[0].keys())
    for tdict in tdicts[1:]:
        assert tdict_keys == set(tdict.keys()), "All TensorDicts must have the same keys"
    return TensorDict(
        {
            k: f(
                *[(a[k] if isinstance(a, TensorDict) else a) for a in args],
                **{ki: (vi[k] if isinstance(vi, TensorDict) else vi) for ki, vi in kwargs.items()},
            )
            for k in tdict_keys
        },
        device=tdicts[0].device,
    ).auto_batch_size_()


def tensordict_cat(tdict_list, dim=0, **kwargs):
    """
    weirdly, the tensordict library requires a strict condition for batch size,
    whereas we just need to concat tensors one by one without needing them to have exact same dimensions.
    """
    return TensorDict(
        dict(
            {
                k: torch.cat([tdict[k] for tdict in tdict_list], dim=dim, **kwargs)
                for k in tdict_list[0].keys()
            }
        ),
        device=tdict_list[0].device,
    ).auto_batch_size_()


def apply_nan_to_num(td):
    return TensorDict(
        {key: torch.nan_to_num(value) for key, value in td.items()}, batch_size=td.batch_size
    )


def apply_isnan(td):
    return TensorDict(
        {key: ~torch.isnan(value) for key, value in td.items()}, batch_size=td.batch_size
    )


def apply_threshold(td):
    return TensorDict(
        {
            key: value.masked_fill(torch.isinf(value) | (value > 1e35), 0)
            for key, value in td.items()
        },
        batch_size=td.batch_size,
    )


def replace_nans(tensordict, value=0):
    return tensordict.apply(
        lambda x: torch.where(torch.isnan(x), torch.tensor(value, dtype=x.dtype), x)
    )
