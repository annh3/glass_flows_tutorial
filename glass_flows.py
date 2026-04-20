#annhe
import torch
from torch import nn, Tensor
from einops import rearrange
from PIL import Image

from flux2.sampling import scatter_ids, compress_time, batched_prc_img, batched_prc_txt, get_schedule


_IMAGE_HEIGHT = 256
_IMAGE_WIDTH = 256
_IMAGE_LATENT_FACTOR = 16
_IMAGE_IN_CHANNELS = 128
_IMAGE_OUT_CHANNELS = 128


def sample_gaussian_latent(batch, height, width, rng=None):
    shape = (batch, _IMAGE_IN_CHANNELS, height // _IMAGE_LATENT_FACTOR, width // _IMAGE_LATENT_FACTOR)
    latent = torch.randn(shape, generator=rng, dtype=torch.bfloat16, device="mps")
    return latent


def format_batch_variable(t, x_t):
    """
    Broadcast timestep t to match the batch size of x_t.
    Returns a tensor of shape (B,) on the same device and dtype as x_t.
    """
    t = torch.tensor(t, device=x_t.device, dtype=x_t.dtype)
    if t.ndim == 0:
        t = t.unsqueeze(0)
    if len(t) < x_t.shape[0]:
        assert len(t) == 1
        t = torch.ones(size=(x_t.shape[0],), device=x_t.device, dtype=x_t.dtype) * t
    return t


def mult_first_dim(x, t):
    """
    Scale each sample x[i] by t[i], broadcasting over non-batch dims.
    """
    if t.ndim == 0:
        return t * x
    t = t.view(-1)
    if x.size(0) != t.size(0):
        raise ValueError("The size of the vector t must match the first dimension of tensor x.")
    t = t.view(-1, *([1] * (x.dim() - 1)))
    return x * t


def grab(x):
    return x.detach().cpu().numpy()


def scatter_ids_differentiable(x, x_ids):
    """
    scatter_ids from the flux library should be differentiable
    but I ran into issues so adding this function in this library file.
    """
    x_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        t_ids_cmpr = compress_time(t_ids)
        t = torch.max(t_ids_cmpr) + 1
        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1
        flat_ids = t_ids_cmpr * w * h + h_ids * w + w_ids

        # root the output tensor in the graph via data
        out = data.new_zeros((t * h * w, ch))
        out = out + data.sum() * 0  # tie to graph without changing values
        out = out.index_put((flat_ids,), data)  # differentiable scatter

        x_list.append(rearrange(out, "(t h w) c -> 1 c t h w", t=t, h=h, w=w))
    return x_list


class FlowModelCFG(nn.Module):
    def __init__(self, flow_model, text_encoder, guidance=4.0):
        super().__init__()
        self.flow_model = flow_model
        self.text_encoder = text_encoder
        self.guidance = guidance

    def forward(self, x_t, timesteps, prompt, **kwargs):
        batch_size, seq_len, channels = x_t.shape
        device = x_t.device
        dtype = x_t.dtype

        h = w = int(seq_len ** 0.5)
        x_ids = torch.cartesian_prod(
            torch.arange(1, device=device),
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            torch.arange(1, device=device),
        ).unsqueeze(0).expand(batch_size, -1, -1)

        prompts = prompt if isinstance(prompt, list) else [prompt]
        ctx_uncond = self.text_encoder([""] * batch_size).to(dtype)
        ctx_cond = self.text_encoder(prompts).to(dtype)
        ctx = torch.cat([ctx_uncond, ctx_cond], dim=0)
        ctx, ctx_ids = batched_prc_txt(ctx)

        x_t_doubled = torch.cat([x_t, x_t], dim=0)
        x_ids_doubled = torch.cat([x_ids, x_ids], dim=0)

        t_tensor = format_batch_variable(timesteps, x_t)           # (B,)
        t_tensor = torch.cat([t_tensor, t_tensor], dim=0)          # (2B,)

        velocity_prediction = self.flow_model.forward(
            x=x_t_doubled, x_ids=x_ids_doubled, timesteps=t_tensor,
            ctx=ctx, ctx_ids=ctx_ids, guidance=None
        )
        velocity_pred_uncond, velocity_pred_cond = velocity_prediction.chunk(2)
        return velocity_pred_uncond + self.guidance * (velocity_pred_cond - velocity_pred_uncond)


def rectified_flow_reverse_process_non_cfg(flow_model, text_encoder, num_denoising_steps, prompt, guidance=4.0, rng=None):
    noise_latent_space = sample_gaussian_latent(1, _IMAGE_HEIGHT, _IMAGE_WIDTH, rng=rng)
    z_t, z_ids = batched_prc_img(noise_latent_space)
    ctx_uncond = text_encoder([""]).to(torch.bfloat16)
    ctx = text_encoder([prompt])
    ctx = torch.cat([ctx_uncond, ctx], dim=0)
    ctx, ctx_ids = batched_prc_txt(ctx)

    schedule = get_schedule(num_denoising_steps, z_t.shape[1])
    z_t = torch.cat([z_t, z_t], dim=0)
    z_ids = torch.cat([z_ids, z_ids], dim=0)

    for t_curr, t_next in zip(schedule[:-1], schedule[1:]):
        t_tensor = torch.tensor([t_curr], device="mps")
        velocity_prediction = flow_model.forward(
            x=z_t, x_ids=z_ids, timesteps=t_tensor, ctx=ctx, ctx_ids=ctx_ids, guidance=None
        )
        velocity_pred_uncond, velocity_pred_cond = velocity_prediction.chunk(2)
        velocity_prediction = velocity_pred_uncond + guidance * (velocity_pred_cond - velocity_pred_uncond)
        velocity_prediction = torch.cat([velocity_prediction, velocity_prediction], dim=0)
        z_t = z_t + (t_next - t_curr) * velocity_prediction

    z_t = z_t.chunk(2)[0]
    z_ids = z_ids.chunk(2)[0]
    return z_t, z_ids


def rectified_flow_reverse_process(flow_model_cfg, num_denoising_steps, prompt, rng=None):
    noise_latent_space = sample_gaussian_latent(1, _IMAGE_HEIGHT, _IMAGE_WIDTH, rng=rng)
    z_t, z_ids = batched_prc_img(noise_latent_space)

    schedule = get_schedule(num_denoising_steps, z_t.shape[1])

    for t_curr, t_next in zip(schedule[:-1], schedule[1:]):
        t_tensor = torch.tensor([t_curr], device="mps")
        velocity_prediction = flow_model_cfg.forward(z_t, t_tensor, prompt)
        z_t = z_t + (t_next - t_curr) * velocity_prediction

    return z_t, z_ids


def decode_and_display(X, X_ids, auto_encoder):
    X_spatial = torch.cat(scatter_ids(X, X_ids)).squeeze(2)
    X_decoded = auto_encoder.decode(X_spatial).float()
    X_decoded = X_decoded.clamp(-1, 1)
    X_decoded = rearrange(X_decoded[0], "c h w -> h w c")
    img_PIL = Image.fromarray((127.5 * (X_decoded + 1.0)).cpu().byte().numpy())
    return img_PIL


class GlassFlow(nn.Module):
    """Wraps a flow matching model and converts it into a posterior flow matching model."""

    def __init__(self,
                 fm_model: nn.Module,
                 clip_val: float = 1e-8,
                 t_min: float = 0.001,
                 t_max: float = 0.999,
                 eta_t_clip: float = 200.0):
        super().__init__()
        self.fm_model = fm_model
        self.clip_val = clip_val
        self.t_min = t_min
        self.t_max = t_max
        self.eta_t_clip = eta_t_clip

    def alpha_t(self, t):
        return t

    def dot_alpha_t(self, t):
        return torch.ones_like(t)

    def sigma_t(self, t):
        return 1 - t

    def dot_sigma_t(self, t):
        return -1 * torch.ones_like(t)

    def g_t(self, t):
        return (self.sigma_t(t) / torch.clip(self.alpha_t(t), min=self.clip_val)) ** 2

    def g_t_inv(self, inp_):
        return 1 / (1 + torch.sqrt(inp_))

    def denoiser(self, x_t, t, **kwargs):
        t = format_batch_variable(t, x_t)
        velocity = -1 * self.fm_model(x_t=x_t, timesteps=1 - t, **kwargs)
        difference = (mult_first_dim(velocity, self.sigma_t(t))
                      - mult_first_dim(x_t, self.dot_sigma_t(t)))
        denominator = (self.dot_alpha_t(t) * self.sigma_t(t)
                       - self.dot_sigma_t(t) * self.alpha_t(t))
        return mult_first_dim(difference, 1 / torch.clip(denominator, min=self.clip_val))

    def bar_alpha_s(self, s: Tensor, bar_alpha_final: float):
        return bar_alpha_final * s

    def dot_bar_alpha_s(self, s: Tensor, bar_alpha_final: float):
        return bar_alpha_final * torch.ones_like(s)

    def bar_sigma_s(self, s: Tensor, sigma_cond_final: float):
        return s * sigma_cond_final + (1 - s)

    def dot_bar_sigma_s(self, s: Tensor, sigma_cond_final: float):
        return torch.ones_like(s) * (sigma_cond_final - 1.0)

    def get_num_stable_inverse(self, matrix: Tensor):
        return torch.linalg.inv(
            matrix + 0.0001 * self.clip_val * torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
        )

    def get_glass_denoiser(self, mu_s: Tensor, Cov_s: Tensor, X_t: Tensor, bar_X_s: Tensor,
                           dtype: torch.dtype, precdtype: torch.dtype, **kwargs):
        inv_Cov_s = self.get_num_stable_inverse(Cov_s)
        bproduct = mu_s @ inv_Cov_s @ mu_s
        t_star = self.g_t_inv(1 / torch.clip(bproduct, min=self.clip_val))

        weights = self.alpha_t(t_star) * (mu_s @ inv_Cov_s) / torch.clip(bproduct, min=self.clip_val)
        scaled_suff_stat = weights[0] * X_t + weights[1] * bar_X_s

        denoiser = self.denoiser(
            x_t=scaled_suff_stat.to(dtype), t=t_star.to(dtype), **kwargs
        ).to(dtype=precdtype)
        return denoiser

    def sample_glass_transition(self,
                                X_t: Tensor,
                                t_start: Tensor,
                                t_end: Tensor,
                                corr_rho: Tensor,
                                n_steps: int,
                                dtype: torch.dtype,
                                device: torch.device,
                                schedule: str = "s_linear",
                                return_traj: bool = False,
                                precdtype: torch.dtype = torch.float64,
                                **kwargs):
        s_vec = torch.linspace(self.t_min, self.t_max, n_steps + 1, dtype=precdtype, device=device)

        alpha_t_start = self.alpha_t(t_start).to(precdtype)
        alpha_t_end = self.alpha_t(t_end).to(precdtype)
        sigma_t_start = self.sigma_t(t_start).to(precdtype)
        sigma_t_end = self.sigma_t(t_end).to(precdtype)

        bar_gamma = corr_rho * sigma_t_end / torch.clip(sigma_t_start, min=self.clip_val)
        bar_X_s_init = bar_gamma * X_t + torch.randn_like(X_t)

        bar_alpha_final = alpha_t_end - bar_gamma * alpha_t_start
        bar_sigma_final = torch.sqrt(torch.clip((sigma_t_end ** 2) * (1 - corr_rho ** 2), min=0.0))

        bar_alpha_s = self.bar_alpha_s(s_vec, bar_alpha_final)
        dot_bar_alpha_s = self.dot_bar_alpha_s(s_vec, bar_alpha_final)
        bar_sigma_s = self.bar_sigma_s(s_vec, bar_sigma_final)
        dot_bar_sigma_s = self.dot_bar_sigma_s(s_vec, bar_sigma_final)

        w_1 = dot_bar_sigma_s / torch.clip(bar_sigma_s, min=self.clip_val)
        w_2 = dot_bar_alpha_s - w_1 * bar_alpha_s
        w_3 = -w_1 * bar_gamma

        X_t = X_t.to(dtype=precdtype)
        bar_X_s = bar_X_s_init

        if return_traj:
            traj_list = [bar_X_s.cpu().detach().float()]

        n_steps = len(s_vec) - 1
        for i in range(n_steps):
            mu_s = torch.tensor(
                [alpha_t_start, bar_alpha_s[i] + bar_gamma * alpha_t_start],
                dtype=precdtype, device=device
            )
            Cov_s = torch.tensor(
                [[sigma_t_start ** 2, bar_gamma * (sigma_t_start ** 2)],
                 [bar_gamma * (sigma_t_start ** 2), bar_sigma_s[i] ** 2 + (bar_gamma ** 2) * (sigma_t_start ** 2)]],
                dtype=precdtype, device=device
            )

            glass_denoiser = self.get_glass_denoiser(
                X_t=X_t, bar_X_s=bar_X_s, mu_s=mu_s, Cov_s=Cov_s,
                dtype=dtype, precdtype=precdtype, **kwargs
            )

            velocity = w_1[i] * bar_X_s + w_2[i] * glass_denoiser + w_3[i] * X_t
            bar_X_s = bar_X_s + (s_vec[i + 1] - s_vec[i]) * velocity

            if return_traj:
                traj_list.append(bar_X_s.cpu().detach())

        if return_traj:
            return traj_list
        return bar_X_s

    def get_ddpm_corr(self, t_start, t_end):
        return (self.alpha_t(t_start) * self.sigma_t(t_end)
                / torch.clip(self.alpha_t(t_end) * self.sigma_t(t_start), min=self.clip_val))

    def sample_glass_transition_ddpm(self, t_start, t_end, **kwargs):
        ddpm_corr = self.get_ddpm_corr(t_start, t_end)
        return self.sample_glass_transition(t_start=t_start, t_end=t_end, corr_rho=ddpm_corr, **kwargs)


class GlassFlowBar_X_sWeighted(GlassFlow):
    def sample_glass_transition(self,
                                X_t: Tensor,
                                t_start: Tensor,
                                t_end: Tensor,
                                corr_rho: Tensor,
                                n_steps: int,
                                dtype: torch.dtype,
                                device: torch.device,
                                schedule: str = "s_linear",
                                return_traj: bool = False,
                                precdtype: torch.dtype = torch.float64,
                                **kwargs):
        prompt = kwargs.get("prompt")
        reward_model = kwargs.get("reward_model")
        auto_encoder = kwargs.get("auto_encoder")
        X_ids = kwargs.get("X_ids")
        prompts = prompt if isinstance(prompt, list) else [prompt]

        s_vec = torch.linspace(self.t_min, self.t_max, n_steps + 1, dtype=precdtype, device=device)

        alpha_t_start = self.alpha_t(t_start).to(precdtype)
        alpha_t_end = self.alpha_t(t_end).to(precdtype)
        sigma_t_start = self.sigma_t(t_start).to(precdtype)
        sigma_t_end = self.sigma_t(t_end).to(precdtype)

        bar_gamma = corr_rho * sigma_t_end / torch.clip(sigma_t_start, min=self.clip_val)
        bar_X_s_init = bar_gamma * X_t + torch.randn_like(X_t)

        bar_alpha_final = alpha_t_end - bar_gamma * alpha_t_start
        bar_sigma_final = torch.sqrt(torch.clip((sigma_t_end ** 2) * (1 - corr_rho ** 2), min=0.0))

        bar_alpha_s = self.bar_alpha_s(s_vec, bar_alpha_final)
        dot_bar_alpha_s = self.dot_bar_alpha_s(s_vec, bar_alpha_final)
        bar_sigma_s = self.bar_sigma_s(s_vec, bar_sigma_final)
        dot_bar_sigma_s = self.dot_bar_sigma_s(s_vec, bar_sigma_final)

        w_1 = dot_bar_sigma_s / torch.clip(bar_sigma_s, min=self.clip_val)
        w_2 = dot_bar_alpha_s - w_1 * bar_alpha_s
        w_3 = -w_1 * bar_gamma

        X_t = X_t.to(dtype=precdtype)
        bar_X_s = bar_X_s_init

        if return_traj:
            traj_list = [bar_X_s.cpu().detach().float()]

        n_steps = len(s_vec) - 1
        for i in range(n_steps):
            mu_s = torch.tensor(
                [alpha_t_start, bar_alpha_s[i] + bar_gamma * alpha_t_start],
                dtype=precdtype, device=device
            )
            Cov_s = torch.tensor(
                [[sigma_t_start ** 2, bar_gamma * (sigma_t_start ** 2)],
                 [bar_gamma * (sigma_t_start ** 2), bar_sigma_s[i] ** 2 + (bar_gamma ** 2) * (sigma_t_start ** 2)]],
                dtype=precdtype, device=device
            )
            with torch.no_grad():
                glass_denoiser = self.get_glass_denoiser(
                    X_t=X_t, bar_X_s=bar_X_s, mu_s=mu_s, Cov_s=Cov_s,
                    dtype=dtype, precdtype=precdtype, **kwargs
                )
                velocity = w_1[i] * bar_X_s + w_2[i] * glass_denoiser + w_3[i] * X_t

            y = bar_X_s.clone().detach().requires_grad_(True)

            with torch.enable_grad():
                X_spatial = torch.cat(scatter_ids_differentiable(y, X_ids)).squeeze(2)
                X_decoded = auto_encoder.decode(X_spatial).float()
                X_decoded_clamped = X_decoded.clamp(-1, 1)
                pixels = (X_decoded_clamped + 1.0) / 2.0
                score = reward_model.score(pixels, prompts).sum()
                reward_gradient = torch.autograd.grad(score, y)[0]

            s_curr = s_vec[i]
            s_next = s_vec[i + 1]

            sigma_s_i = self.sigma_t(s_vec[i])
            alpha_s_i = self.alpha_t(s_vec[i])
            weight = sigma_s_i ** 2 / torch.clip(alpha_s_i, min=self.clip_val)

            bar_X_s = bar_X_s + (s_next - s_curr) * (velocity + weight * reward_gradient)

            if return_traj:
                traj_list.append(bar_X_s.cpu().detach())

        if return_traj:
            return traj_list
        return bar_X_s
