"""
Training script for Flux model using DiffusionDB dataset.

This implements flow matching training where:
- z_t = (1 - t) * noise + t * z_clean
- velocity v = z_clean - noise
- Loss = MSE(v_pred, v)

Usage:
    python -m jflux.train --subset "2m_first_1k" --num_epochs 10 --batch_size 4
"""

import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import optax
from chex import Array, PRNGKey
from datasets import load_dataset
from einops import rearrange, repeat
from fire import Fire
from flax import nnx
from PIL import Image
from tqdm import tqdm

from jflux.model import Flux, FluxParams
from jflux.modules.autoencoder import AutoEncoder
from jflux.modules.conditioner import HFEmbedder
from jflux.modules.layers import timestep_embedding
from jflux.util import configs, load_ae, load_clip, load_flow_model, load_t5, torch2jax


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@dataclass
class TrainConfig:
    """Training configuration."""

    # Dataset
    subset: str = "2m_first_1k"  # DiffusionDB subset: 2m_first_1k, 2m_first_10k, etc.
    image_size: int = 256  # Target image size (must be multiple of 16)
    
    # Model
    model_name: str = "flux-dev"  # flux-dev or flux-schnell
    
    # Training
    num_epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    
    # Checkpointing
    save_every: int = 1000
    output_dir: str = "./checkpoints"
    
    # Hardware
    seed: int = 42
    mixed_precision: bool = True


def preprocess_image(
    image: Image.Image, target_size: int = 256
) -> np.ndarray:
    """Preprocess image to target size and normalize to [-1, 1]."""
    # Resize maintaining aspect ratio and center crop
    w, h = image.size
    scale = target_size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    
    # Center crop
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    image = image.crop((left, top, left + target_size, top + target_size))
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Normalize to [-1, 1]
    arr = np.array(image, dtype=np.float32) / 127.5 - 1.0
    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))
    return arr


# Mapping from subset names to number of samples
DIFFUSIONDB_SUBSETS = {
    "2m_first_1k": 1000,
    "2m_first_5k": 5000,
    "2m_first_10k": 10000,
    "2m_first_50k": 50000,
    "2m_first_100k": 100000,
    "2m_random_1k": 1000,
    "2m_random_5k": 5000,
    "2m_random_10k": 10000,
    "2m_random_50k": 50000,
    "2m_random_100k": 100000,
}


class DiffusionDBDataLoader:
    """DataLoader for DiffusionDB dataset from Hugging Face."""
    
    def __init__(
        self,
        subset: str = "2m_first_1k",
        batch_size: int = 4,
        image_size: int = 256,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize DiffusionDB dataloader.
        
        Args:
            subset: DiffusionDB subset to use. Options include:
                - "2m_first_1k": First 1K images (good for testing)
                - "2m_first_5k": First 5K images
                - "2m_first_10k": First 10K images
                - "2m_first_50k": First 50K images
                - "2m_first_100k": First 100K images
                - "2m_random_1k" to "2m_random_100k": Random subsets
        """
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        
        print(f"Loading DiffusionDB subset: {subset}")
        
        # Get number of samples for this subset
        num_samples = DIFFUSIONDB_SUBSETS.get(subset, 1000)
        
        # Load from parquet files (new format)
        # DiffusionDB stores data in numbered parquet parts
        try:
            # Try loading as standard dataset first (works for converted datasets)
            self.dataset = load_dataset(
                "poloclub/diffusiondb",
                "2m_random_1k",  # Use the parquet-based config
                split="train",
            )
            # Limit to requested subset size
            if len(self.dataset) > num_samples:
                indices = list(range(num_samples))
                self.dataset = self.dataset.select(indices)
        except Exception as e:
            print(f"Standard loading failed: {e}")
            print("Trying alternative loading method...")
            # Alternative: load from parquet files directly
            self.dataset = load_dataset(
                "parquet",
                data_files=f"hf://datasets/poloclub/diffusiondb/*/part-*.parquet",
                split="train",
            )
            # Limit to requested subset size
            if len(self.dataset) > num_samples:
                if self.shuffle:
                    indices = self.rng.choice(len(self.dataset), num_samples, replace=False).tolist()
                else:
                    indices = list(range(num_samples))
                self.dataset = self.dataset.select(indices)
        
        print(f"Loaded {len(self.dataset)} samples")
        
    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size
    
    def __iter__(self) -> Iterator[dict[str, Any]]:
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(indices)
        
        for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch = self.dataset.select(batch_indices.tolist())
            
            images = []
            prompts = []
            
            for item in batch:
                img = preprocess_image(item["image"], self.image_size)
                images.append(img)
                prompts.append(item["prompt"])
            
            yield {
                "images": np.stack(images, axis=0),
                "prompts": prompts,
            }


def create_img_ids(height: int, width: int, batch_size: int) -> Array:
    """Create image position IDs for the transformer."""
    h, w = height // 2, width // 2  # After patchifying
    img_ids = jnp.zeros((h, w, 3))
    img_ids = img_ids.at[..., 1].set(jnp.arange(h)[:, None])
    img_ids = img_ids.at[..., 2].set(jnp.arange(w)[None, :])
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
    return img_ids


def prepare_latents(
    ae: AutoEncoder,
    images: Array,
) -> Array:
    """Encode images to latent space using the VAE."""
    # images: (B, C, H, W) in [-1, 1]
    latents = ae.encode(images)
    return latents


def patchify(latents: Array) -> Array:
    """Convert latents to patch sequences for the transformer."""
    # latents: (B, C, H, W) -> (B, H*W/4, C*4)
    return rearrange(latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


def unpatchify(patches: Array, height: int, width: int) -> Array:
    """Convert patch sequences back to latent images."""
    return rearrange(
        patches,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=height // 2 // 2,
        w=width // 2 // 2,
        ph=2,
        pw=2,
    )


def flow_matching_loss(
    model: Flux,
    latents: Array,  # Clean latents (B, seq, dim)
    img_ids: Array,
    txt: Array,
    txt_ids: Array,
    vec: Array,
    timesteps: Array,  # (B,)
    noise: Array,  # Same shape as latents
    guidance: Array | None = None,
) -> Array:
    """
    Compute flow matching loss.
    
    Flow matching interpolates between noise and data:
        z_t = (1 - t) * noise + t * z_clean
    
    The velocity is:
        v = z_clean - noise
    
    Loss is MSE between predicted and target velocity.
    """
    # Expand timesteps for broadcasting: (B,) -> (B, 1, 1)
    t = timesteps[:, None, None]
    
    # Interpolate: z_t = (1 - t) * noise + t * z_clean
    z_t = (1.0 - t) * noise + t * latents
    
    # Target velocity: v = z_clean - noise
    v_target = latents - noise
    
    # Predict velocity
    v_pred = model(
        img=z_t,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        y=vec,
        timesteps=timesteps,
        guidance=guidance,
    )
    
    # MSE loss
    loss = jnp.mean((v_pred - v_target) ** 2)
    return loss


def create_train_state(
    model: Flux,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    total_steps: int,
    gradient_clip: float,
) -> tuple[nnx.Optimizer, optax.GradientTransformation]:
    """Create optimizer and learning rate schedule."""
    # Learning rate schedule: linear warmup then cosine decay
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_steps,
    )
    decay_fn = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=total_steps - warmup_steps,
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[warmup_steps],
    )
    
    # Optimizer with gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(gradient_clip),
        optax.adamw(learning_rate=schedule_fn, weight_decay=weight_decay),
    )
    
    optimizer = nnx.Optimizer(model, tx)
    return optimizer, tx


@nnx.jit
def train_step(
    model: Flux,
    optimizer: nnx.Optimizer,
    img: Array,
    img_ids: Array,
    txt: Array,
    txt_ids: Array,
    vec: Array,
    timesteps: Array,
    noise: Array,
    guidance: Array | None,
) -> Array:
    """Single training step."""
    
    def loss_fn(model):
        return flow_matching_loss(
            model=model,
            latents=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            vec=vec,
            timesteps=timesteps,
            noise=noise,
            guidance=guidance,
        )
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss


def save_checkpoint(
    model: Flux,
    optimizer: nnx.Optimizer,
    step: int,
    output_dir: str,
):
    """Save model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"checkpoint_{step}")
    
    # Save model state
    state = nnx.state(model)
    # Use orbax or simple pickle for checkpointing
    # For simplicity, we'll use nnx's built-in serialization
    import pickle
    with open(f"{path}_model.pkl", "wb") as f:
        pickle.dump(state, f)
    
    print(f"Saved checkpoint to {path}")


def train(
    # Dataset
    subset: str = "2m_first_1k",
    image_size: int = 256,
    # Model
    model_name: str = "flux-dev",
    # Training
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    gradient_clip: float = 1.0,
    # Checkpointing
    save_every: int = 1000,
    output_dir: str = "./checkpoints",
    # Hardware
    seed: int = 42,
    freeze_text_encoders: bool = True,
    freeze_vae: bool = True,
):
    """
    Train Flux model on DiffusionDB dataset.
    
    Args:
        subset: DiffusionDB subset. Options:
            - "2m_first_1k": 1K images (testing)
            - "2m_first_5k": 5K images
            - "2m_first_10k": 10K images
            - "2m_random_50k": 50K random images
            - "large_random_10k": 10K high-res images
        image_size: Target image size (multiple of 16)
        model_name: "flux-dev" or "flux-schnell"
        num_epochs: Number of training epochs
        batch_size: Batch size (reduce for memory)
        learning_rate: Peak learning rate
        weight_decay: AdamW weight decay
        warmup_steps: LR warmup steps
        gradient_clip: Max gradient norm
        save_every: Save checkpoint every N steps
        output_dir: Checkpoint directory
        seed: Random seed
        freeze_text_encoders: Keep T5/CLIP frozen (recommended)
        freeze_vae: Keep VAE frozen (recommended)
    """
    # Detect device
    devices = jax.devices()
    device_info = devices[0]
    device_type = device_info.platform.upper()
    device_name = getattr(device_info, 'device_kind', device_type)
    
    print("=" * 60)
    print("Flux Training Script")
    print("=" * 60)
    print(f"Device: {device_name} ({device_type})")
    print(f"Number of devices: {len(devices)}")
    print(f"Dataset: DiffusionDB ({subset})")
    print(f"Model: {model_name}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print("=" * 60)
    
    # Set random seeds
    key = jax.random.PRNGKey(seed)
    np.random.seed(seed)
    
    # Ensure image size is valid
    assert image_size % 16 == 0, "Image size must be multiple of 16"
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    dataloader = DiffusionDBDataLoader(
        subset=subset,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
    )
    
    # Calculate total steps
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * num_epochs
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    
    # Load models
    print("\n[2/4] Loading models...")
    # Use the detected device type
    jax_device = device_type.lower()  # "gpu", "tpu", or "cpu"
    torch_device = "cuda" if jax_device == "gpu" else "cpu"
    
    print(f"  JAX device: {jax_device}")
    print(f"  PyTorch device: {torch_device}")
    print("  Loading T5 encoder...")
    t5 = load_t5(device=torch_device, max_length=512)
    print("  Loading CLIP encoder...")
    clip = load_clip(device=torch_device)
    print("  Loading VAE...")
    ae = load_ae(model_name, device=jax_device)
    print("  Loading Flux model...")
    model = load_flow_model(model_name, device=jax_device)
    
    # Pre-compute image IDs (same for all batches with same size)
    img_ids = create_img_ids(
        height=image_size // 8,  # VAE downsamples by 8
        width=image_size // 8,
        batch_size=batch_size,
    )
    
    # Create optimizer
    print("\n[3/4] Setting up optimizer...")
    optimizer, _ = create_train_state(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        gradient_clip=gradient_clip,
    )
    
    # Training loop
    print("\n[4/4] Starting training...")
    global_step = 0
    use_guidance = model_name == "flux-dev"
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            total=len(dataloader),
        )
        
        for batch in progress:
            # Prepare data
            images = jnp.array(batch["images"])
            prompts = batch["prompts"]
            
            # Encode images to latents
            latents = prepare_latents(ae, images)
            latents_seq = patchify(latents)
            
            # Get text embeddings
            txt = torch2jax(t5(prompts))
            vec = torch2jax(clip(prompts))
            txt_ids = jnp.zeros((batch_size, txt.shape[1], 3))
            
            # Sample random timesteps
            key, subkey = jax.random.split(key)
            timesteps = jax.random.uniform(subkey, (batch_size,), minval=0.0, maxval=1.0)
            
            # Sample noise
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, latents_seq.shape, dtype=latents_seq.dtype)
            
            # Guidance (only for flux-dev)
            guidance = jnp.full((batch_size,), 4.0) if use_guidance else None
            
            # Training step
            loss = train_step(
                model=model,
                optimizer=optimizer,
                img=latents_seq,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                vec=vec,
                timesteps=timesteps,
                noise=noise,
                guidance=guidance,
            )
            
            epoch_loss += float(loss)
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            progress.set_postfix({
                "loss": f"{float(loss):.4f}",
                "avg_loss": f"{epoch_loss / num_batches:.4f}",
            })
            
            # Save checkpoint
            if global_step % save_every == 0:
                save_checkpoint(model, optimizer, global_step, output_dir)
        
        # End of epoch summary
        avg_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, global_step, output_dir)
    print("\nTraining completed!")


def app():
    """CLI entry point."""
    Fire(train)


if __name__ == "__main__":
    app()
