import warnings

import pytest
import torch
from tensordict.tensordict import TensorDict

from geoarches.utils.tensordict_utils import apply_mask_from_gt_nans, check_pred_has_no_nans


class TestCheckPredHasNoNans:
    def test_no_nans(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        with warnings.catch_warnings(record=True) as w:
            check_pred_has_no_nans(pred, target)
            assert len(w) == 0

    def test_target_nans_only(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, float("nan"), 3.0])
        with warnings.catch_warnings(record=True) as w:
            check_pred_has_no_nans(pred, target)
            assert len(w) == 0

    def test_all_nans_target(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([float("nan"), float("nan"), float("nan")])
        with warnings.catch_warnings(record=True) as w:
            check_pred_has_no_nans(pred, target)
            assert len(w) == 0

    def test_pred_has_nans_where_target_has_nans(self):
        pred = torch.tensor([1.0, float("nan"), 3.0])
        target = torch.tensor([1.0, float("nan"), 3.0])
        with pytest.warns(UserWarning, match="Prediction has NaNs where target data has NaNs"):
            check_pred_has_no_nans(pred, target)

    def test_all_nans_pred_and_target(self):
        pred = torch.tensor([float("nan"), float("nan"), float("nan")])
        target = torch.tensor([float("nan"), float("nan"), float("nan")])
        with pytest.warns(UserWarning, match="Prediction has NaNs where target data has NaNs"):
            check_pred_has_no_nans(pred, target)

    def test_pred_has_nans_where_target_has_no_nans(self):
        pred = torch.tensor([1.0, float("nan"), 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        with pytest.warns(UserWarning, match="Prediction has NaNs where target data has no NaNs"):
            check_pred_has_no_nans(pred, target)

    def test_mixed_nans(self):
        pred = torch.tensor([float("nan"), float("nan"), 3.0])
        target = torch.tensor([1.0, float("nan"), 3.0])
        with warnings.catch_warnings(record=True) as w:
            check_pred_has_no_nans(pred, target)
            assert len(w) == 2
            assert any(
                "Prediction has NaNs where target data has no NaNs" in str(warning.message)
                for warning in w
            )
            assert any(
                "Prediction has NaNs where target data has NaNs" in str(warning.message)
                for warning in w
            )


class TestApplyMaskFromGtNans:
    def test_no_nans_in_gt(self):
        pred = TensorDict({"a": torch.tensor([1.0, 2.0, 3.0])}, batch_size=[3])
        gt = TensorDict({"a": torch.tensor([4.0, 5.0, 6.0])}, batch_size=[3])
        fill_value = 0.0
        new_pred, new_gt = apply_mask_from_gt_nans(pred.clone(), gt.clone(), fill_value)

        assert torch.equal(new_pred["a"], pred["a"])
        assert torch.equal(new_gt["a"], gt["a"])

    def test_nans_in_gt(self):
        pred = TensorDict({"a": torch.tensor([1.0, 2.0, 3.0])}, batch_size=[3])
        gt = TensorDict({"a": torch.tensor([4.0, float("nan"), 6.0])}, batch_size=[3])
        fill_value = -1.0
        new_pred, new_gt = apply_mask_from_gt_nans(pred.clone(), gt.clone(), fill_value)

        expected_pred = torch.tensor([1.0, -1.0, 3.0])
        expected_gt = torch.tensor([4.0, -1.0, 6.0])

        assert torch.equal(new_pred["a"], expected_pred)
        assert torch.equal(new_gt["a"], expected_gt)

    def test_nans_in_gt_multiple_keys(self):
        pred = TensorDict(
            {"a": torch.tensor([1.0, 2.0, 3.0]), "b": torch.tensor([7.0, 8.0, 9.0])},
            batch_size=[3],
        )
        gt = TensorDict(
            {
                "a": torch.tensor([4.0, float("nan"), 6.0]),
                "b": torch.tensor([float("nan"), 11.0, float("nan")]),
            },
            batch_size=[3],
        )
        expected_pred_a = torch.tensor([1.0, 0.0, 3.0])
        expected_pred_b = torch.tensor([0.0, 8.0, 0.0])
        expected_gt_a = torch.tensor([4.0, 0.0, 6.0])
        expected_gt_b = torch.tensor([0.0, 11.0, 0.0])

        fill_value = 0.0
        new_pred, new_gt = apply_mask_from_gt_nans(pred.clone(), gt.clone(), fill_value)

        assert torch.equal(new_pred["a"], expected_pred_a)
        assert torch.equal(new_gt["a"], expected_gt_a)
        assert torch.equal(new_pred["b"], expected_pred_b)
        assert torch.equal(new_gt["b"], expected_gt_b)

    def test_fill_value_is_nan(self):
        pred = TensorDict({"a": torch.tensor([1.0, 2.0, 3.0])}, batch_size=[3])
        gt = TensorDict({"a": torch.tensor([4.0, float("nan"), 6.0])}, batch_size=[3])
        fill_value = float("nan")
        new_pred, new_gt = apply_mask_from_gt_nans(pred.clone(), gt.clone(), fill_value)

        # Cannot use torch.equal since NaNs are never equal to each other.
        # Pred and GT are NaN where GT is NaN.
        assert torch.all(torch.isnan(new_gt["a"]) == torch.isnan(gt["a"]))
        assert torch.all(torch.isnan(new_pred["a"]) == torch.isnan(gt["a"]))
        # Pred and GT are not NaN where GT is not NaN.
        assert torch.equal(new_gt["a"][~torch.isnan(gt["a"])], gt["a"][~torch.isnan(gt["a"])])
        assert torch.equal(new_pred["a"][~torch.isnan(gt["a"])], pred["a"][~torch.isnan(gt["a"])])

    def test_all_nans_in_gt(self):
        pred = TensorDict({"a": torch.tensor([1.0, 2.0, 3.0])}, batch_size=[3])
        gt = TensorDict(
            {"a": torch.tensor([float("nan"), float("nan"), float("nan")])},
            batch_size=[3],
        )
        fill_value = 5.0
        new_pred, new_gt = apply_mask_from_gt_nans(pred.clone(), gt.clone(), fill_value)

        expected_tensor = torch.tensor([5.0, 5.0, 5.0])
        assert torch.equal(new_pred["a"], expected_tensor)
        assert torch.equal(new_gt["a"], expected_tensor)
