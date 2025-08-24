import warnings

import torch
from tensordict.tensordict import TensorDict


def apply_mask_from_gt_nans(pred: TensorDict, ground_truth: TensorDict, value) -> TensorDict:
    """
    Applies a mask retrieved from the ground truth to the predictions.
    The mask is created by checking where the ground truth has NaNs.
    The predictions are then multiplied by the binary mask accordingly.

    This function is useful to ensure that variables like sea_ice_cover are treated correctly,
    where the ground truth may have NaNs in certain areas, and we want to apply a mask to the predictions
    to avoid using those NaNs in the loss calculation.
    """

    # Mask predictions with binary mask from ground truth
    # where ground truth is not NaN, the mask is 1, otherwise 0

    # pred = TensorDict(
    #    {k: (~torch.isnan(ground_truth[k])).float()  * v for k, v in pred.items()}, batch_size=pred.batch_size
    # )
    value = torch.tensor(value)

    for k, v in ground_truth.items():
        gt_valid = ~torch.isnan(v)
        pred[k] = pred[k].where(gt_valid, value)
        ground_truth[k] = ground_truth[k].where(gt_valid, value)

    return pred, ground_truth


def check_pred_has_no_nans(pred: torch.Tensor, target: torch.Tensor):
    """
    Pred is a tensor with predictions.
    Target is a tensor with targets.
    The function checks if pred has no NaNs where target has no NaNs.
    """

    target_valid = ~target.isnan()
    target_nans = target.isnan()

    # index pred with target_nans to check if pred has no NaNs where target has no NaNs
    pred_should_be_valid = pred.where(target_valid, 0)
    if pred_should_be_valid.isnan().any():
        warnings.warn("Prediction has NaNs where target data has no NaNs")

    pred_where_target_has_nans = pred.where(target_nans, 0)
    if pred_where_target_has_nans.isnan().any():
        warnings.warn("Prediction has NaNs where target data has NaNs")

    return pred


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
