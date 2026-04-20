"""
Unit tests for scatter_ids_differentiable vs scatter_ids (flux2).

Run with:
    conda run -n glass_flows python -m pytest glass_flows_tests.py -v
"""
import torch
import pytest
from flux2.sampling import scatter_ids, batched_prc_img
from glass_flows import scatter_ids_differentiable, sample_gaussian_latent, _IMAGE_HEIGHT, _IMAGE_WIDTH


def _make_packed_input(batch_size=1, seed=42):
    torch.manual_seed(seed)
    noise = sample_gaussian_latent(batch_size, _IMAGE_HEIGHT, _IMAGE_WIDTH)
    x_packed, x_ids = batched_prc_img(noise)
    return x_packed, x_ids


class TestScatterIdsDifferentiable:
    def test_output_shape_matches_scatter_ids(self):
        x, x_ids = _make_packed_input()
        ref = scatter_ids(x, x_ids)
        out = scatter_ids_differentiable(x, x_ids)
        assert len(out) == len(ref)
        for r, o in zip(ref, out):
            assert r.shape == o.shape, f"shape mismatch: ref={r.shape} diff={o.shape}"

    def test_values_match_scatter_ids(self):
        x, x_ids = _make_packed_input()
        ref = scatter_ids(x, x_ids)
        out = scatter_ids_differentiable(x, x_ids)
        for i, (r, o) in enumerate(zip(ref, out)):
            assert torch.allclose(r.float(), o.float(), atol=1e-5), \
                f"item {i}: max diff = {(r.float() - o.float()).abs().max()}"

    def test_batch_output_shape_matches_scatter_ids(self):
        x, x_ids = _make_packed_input(batch_size=2)
        ref = scatter_ids(x, x_ids)
        out = scatter_ids_differentiable(x, x_ids)
        assert len(out) == len(ref)
        for r, o in zip(ref, out):
            assert r.shape == o.shape

    def test_batch_values_match_scatter_ids(self):
        x, x_ids = _make_packed_input(batch_size=2)
        ref = scatter_ids(x, x_ids)
        out = scatter_ids_differentiable(x, x_ids)
        for i, (r, o) in enumerate(zip(ref, out)):
            assert torch.allclose(r.float(), o.float(), atol=1e-5), \
                f"batch item {i}: max diff = {(r.float() - o.float()).abs().max()}"

    def test_gradient_flows_through(self):
        x, x_ids = _make_packed_input()
        x = x.float().requires_grad_(True)
        out = scatter_ids_differentiable(x, x_ids)
        loss = torch.cat(out).sum()
        loss.backward()
        assert x.grad is not None, "gradient did not flow back to input"
        assert x.grad.shape == x.shape
        assert not torch.all(x.grad == 0), "all-zero gradient — backward may be broken"

    def test_output_is_list(self):
        x, x_ids = _make_packed_input()
        out = scatter_ids_differentiable(x, x_ids)
        assert isinstance(out, list)

    def test_output_spatial_dims_5d(self):
        x, x_ids = _make_packed_input()
        out = scatter_ids_differentiable(x, x_ids)
        for tensor in out:
            assert tensor.ndim == 5, f"expected 5D (1 C T H W), got {tensor.ndim}D"
            assert tensor.shape[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
