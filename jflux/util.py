import os
from dataclasses import dataclass

import jax
import torch  # need for torch 2 jax
from chex import Array
from flax import nnx
from huggingface_hub import hf_hub_download
from jax import numpy as jnp
from safetensors import safe_open

from jflux.model import Flux, FluxParams
from jflux.modules.autoencoder import AutoEncoder, AutoEncoderParams
from jflux.modules.conditioner import HFEmbedder
from jflux.port import port_autoencoder, port_flux


def torch2jax(torch_tensor: torch.Tensor) -> Array:
    is_bfloat16 = torch_tensor.dtype == torch.bfloat16
    if is_bfloat16:
        # upcast the tensor to fp32
        torch_tensor = torch_tensor.to(dtype=torch.float32)

    if torch.device.type != "cpu":
        torch_tensor = torch_tensor.to("cpu")

    numpy_value = torch_tensor.numpy()
    jax_array = jnp.array(numpy_value, dtype=jnp.bfloat16 if is_bfloat16 else None)
    return jax_array


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
            rngs=nnx.Rngs(default=42),
            param_dtype=jnp.bfloat16,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
            rngs=nnx.Rngs(default=42),
            param_dtype=jnp.bfloat16,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
            rngs=nnx.Rngs(default=42),
            param_dtype=jnp.bfloat16,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
            rngs=nnx.Rngs(default=42),
            param_dtype=jnp.bfloat16,
        ),
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


# Model scale configurations for training from scratch
# These reduce model size for memory-constrained environments
MODEL_SCALES = {
    "tiny": {
        "depth": 2,
        "depth_single_blocks": 4,
        "hidden_size": 384,
        "num_heads": 6,
        "description": "~50M params, requires ~2GB GPU memory",
    },
    "small": {
        "depth": 4,
        "depth_single_blocks": 8,
        "hidden_size": 768,
        "num_heads": 12,
        "description": "~400M params, requires ~8GB GPU memory",
    },
    "base": {
        "depth": 8,
        "depth_single_blocks": 16,
        "hidden_size": 1536,
        "num_heads": 24,
        "description": "~2B params, requires ~24GB GPU memory",
    },
    "full": {
        "depth": 19,
        "depth_single_blocks": 38,
        "hidden_size": 3072,
        "num_heads": 24,
        "description": "~12B params (original), requires ~96GB GPU memory",
    },
}


def load_flow_model(
    name: str,
    device: str,
    hf_download: bool = True,
    from_scratch: bool = False,
    context_in_dim: int | None = None,
    model_scale: str = "full",
) -> Flux:
    """Load Flux model.
    
    Args:
        name: Model config name (flux-dev or flux-schnell)
        device: Device to load on (gpu, tpu, cpu)
        hf_download: Whether to download weights from HuggingFace
        from_scratch: If True, initialize with random weights (for training from scratch)
        context_in_dim: Override context_in_dim (T5 output dim) for training from scratch
        model_scale: Model scale for training from scratch: tiny (~50M), small (~400M), 
                    base (~2B), or full (~12B). Only used when from_scratch=True.
    """
    from dataclasses import replace
    
    device = jax.devices(device)[0]
    with jax.default_device(device):
        params = configs[name].params
        
        # Apply model scale if training from scratch
        if from_scratch and model_scale != "full":
            if model_scale not in MODEL_SCALES:
                raise ValueError(f"Unknown model_scale: {model_scale}. Choose from: {list(MODEL_SCALES.keys())}")
            
            scale_config = MODEL_SCALES[model_scale]
            print(f"  Using '{model_scale}' scale: {scale_config['description']}")
            params = FluxParams(
                in_channels=params.in_channels,
                vec_in_dim=params.vec_in_dim,
                context_in_dim=context_in_dim if context_in_dim else params.context_in_dim,
                hidden_size=scale_config["hidden_size"],
                mlp_ratio=params.mlp_ratio,
                num_heads=scale_config["num_heads"],
                depth=scale_config["depth"],
                depth_single_blocks=scale_config["depth_single_blocks"],
                axes_dim=params.axes_dim,
                theta=params.theta,
                qkv_bias=params.qkv_bias,
                guidance_embed=params.guidance_embed,
                rngs=params.rngs,
                param_dtype=params.param_dtype,
            )
        # Override context_in_dim if specified (for different T5 sizes)
        elif context_in_dim is not None and from_scratch:
            params = FluxParams(
                in_channels=params.in_channels,
                vec_in_dim=params.vec_in_dim,
                context_in_dim=context_in_dim,
                hidden_size=params.hidden_size,
                mlp_ratio=params.mlp_ratio,
                num_heads=params.num_heads,
                depth=params.depth,
                depth_single_blocks=params.depth_single_blocks,
                axes_dim=params.axes_dim,
                theta=params.theta,
                qkv_bias=params.qkv_bias,
                guidance_embed=params.guidance_embed,
                rngs=params.rngs,
                param_dtype=params.param_dtype,
            )
            print(f"  Using custom context_in_dim={context_in_dim} for T5 encoder")
        
        if from_scratch:
            print(f"Initializing Flux with random weights on {device}")
            model = Flux(params=params)
            return model
        
        ckpt_path = configs[name].ckpt_path
        if (
            ckpt_path is None
            and configs[name].repo_id is not None
            and configs[name].repo_flow is not None
            and hf_download
        ):
            ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

        print(f"Load and port flux on {device}")

        model = Flux(params=params)
        if ckpt_path is not None:
            tensors = {}
            with safe_open(ckpt_path, framework="pt") as f:
                for k in f.keys():
                    tensors[k] = torch2jax(f.get_tensor(k))

            model = port_flux(flux=model, tensors=tensors)

            del tensors
            jax.clear_caches()
    return model


# T5 model options (smaller = less disk space, faster, but lower quality)
T5_MODELS = {
    "xxl": {"name": "ariG23498/t5-v1_1-xxl-torch", "dim": 4096},  # 9.5GB - best quality
    "xl": {"name": "google/t5-v1_1-xl", "dim": 2048},              # 3GB
    "large": {"name": "google/t5-v1_1-large", "dim": 1024},        # 800MB
    "base": {"name": "google/t5-v1_1-base", "dim": 768},           # 250MB - fastest
}


def get_t5_dim(model_size: str) -> int:
    """Get the output dimension for a T5 model size."""
    if model_size not in T5_MODELS:
        raise ValueError(f"Unknown T5 size: {model_size}. Choose from {list(T5_MODELS.keys())}")
    return T5_MODELS[model_size]["dim"]


def load_t5(
    device: str | torch.device = "cuda",
    max_length: int = 512,
    model_size: str = "xxl",
) -> HFEmbedder:
    """Load T5 text encoder.
    
    Args:
        device: Device to load on
        max_length: Max sequence length (64, 128, 256, 512)
        model_size: T5 model size - "xxl" (9.5GB), "xl" (3GB), "large" (800MB), "base" (250MB)
    """
    if model_size not in T5_MODELS:
        raise ValueError(f"Unknown T5 size: {model_size}. Choose from {list(T5_MODELS.keys())}")
    
    model_name = T5_MODELS[model_size]["name"]
    print(f"  Loading T5 model: {model_name}")
    return HFEmbedder(
        model_name, max_length=max_length, torch_dtype=torch.bfloat16
    ).to(device)


def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    return HFEmbedder(
        "ariG23498/clip-vit-large-patch14-torch",
        max_length=77,
        torch_dtype=torch.bfloat16,
    ).to(device)


def load_ae(name: str, device: str, hf_download: bool = True) -> AutoEncoder:
    device = jax.devices(device)[0]
    with jax.default_device(device):
        ckpt_path = configs[name].ae_path
        if (
            ckpt_path is None
            and configs[name].repo_id is not None
            and configs[name].repo_ae is not None
            and hf_download
        ):
            ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae)

        print(f"Load and port autoencoder on {device}")
        ae = AutoEncoder(params=configs[name].ae_params)

        if ckpt_path is not None:
            tensors = {}
            with safe_open(ckpt_path, framework="pt") as f:
                for k in f.keys():
                    tensors[k] = torch2jax(f.get_tensor(k))
            ae = port_autoencoder(autoencoder=ae, tensors=tensors)

            del tensors
            jax.clear_caches()
    return ae
