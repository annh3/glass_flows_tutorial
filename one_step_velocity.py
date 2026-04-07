"""
One-step velocity prediction from a FLUX.2 flow model.

Only dependency: torch.

Usage:
    wrapper = FlowModelWrapper(fm_model, text_encoder)
    velocity = wrapper(x_t=x_t, x_ids=x_ids, t=t, text_prompts=prompts)

    x_t    : noisy image tokens  [B, L, 128]
    x_ids  : position ids        [B, L, 4]   (build with prc_img from flux2.sampling)
    t      : timestep            [B]          values in [0, 1]
    prompts: list[str] of length B
"""

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Inlined from flux2.sampling (no other deps needed)
# ---------------------------------------------------------------------------

def _prc_txt(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    _l, _ = x.shape
    coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(1),
        "w": torch.arange(1),
        "l": torch.arange(_l),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    return x, x_ids.to(x.device)


def _batched_prc_txt(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    results = [_prc_txt(x[i], t_coord[i] if t_coord is not None else None) for i in range(len(x))]
    xs, x_ids = zip(*results)
    return torch.stack(xs), torch.stack(x_ids)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def one_step_velocity(
    fm_model,
    text_encoder,
    x_t: Tensor,
    x_ids: Tensor,
    t: Tensor,
    text_prompts: list[str],
    guidance: float = 1.0,
) -> Tensor:
    device, dtype = x_t.device, x_t.dtype

    txt_embeds = text_encoder(text_prompts)
    txt_tokens, txt_ids = _batched_prc_txt(txt_embeds)
    txt_tokens = txt_tokens.to(device=device, dtype=dtype)
    txt_ids = txt_ids.to(device=device)

    guidance_vec = torch.full((x_t.shape[0],), guidance, device=device, dtype=dtype)

    return fm_model(
        x=x_t,
        x_ids=x_ids,
        timesteps=t,
        ctx=txt_tokens,
        ctx_ids=txt_ids,
        guidance=guidance_vec,
    )


class FlowModelWrapper:
    def __init__(self, fm_model, text_encoder):
        self.fm_model = fm_model
        self.text_encoder = text_encoder

    def __call__(
        self,
        x_t: Tensor,
        x_ids: Tensor,
        t: Tensor,
        text_prompts: list[str],
        guidance: float = 1.0,
    ) -> Tensor:
        return one_step_velocity(
            self.fm_model, self.text_encoder,
            x_t=x_t, x_ids=x_ids, t=t,
            text_prompts=text_prompts, guidance=guidance,
        )

