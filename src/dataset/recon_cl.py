import os
import random
import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from diffusers import DiffusionPipeline, StableDiffusion3Pipeline
from diffusers.image_processor import VaeImageProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def retrieve_latents(encoder_output, generator=None):
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.sample(generator=generator)
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("encoder_output has neither latent_dist nor latents")


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class RealDataset(Dataset):                          # FIX: was nn.Module, should be Dataset
    def __init__(self, is_train, args):
        root = args.data_path if is_train else args.eval_data_path
        self.data_list = []

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        for filename in os.listdir(root):
            file_path = os.path.join(root, filename)
            if not os.path.isdir(file_path):
                continue

            if '0_real' not in os.listdir(file_path):
                # nested: root/filename/folder_name/0_real/
                for folder_name in os.listdir(file_path):
                    folder_path = os.path.join(file_path, folder_name)
                    if not os.path.isdir(folder_path):
                        continue
                    assert set(os.listdir(folder_path)) == {'0_real'}, \
                        f"Unexpected structure in {folder_path}"
                    for image_name in os.listdir(os.path.join(folder_path, '0_real')):
                        self.data_list.append({
                            "image_path": os.path.join(folder_path, '0_real', image_name),
                            "label": 0
                        })
            else:
                # flat: root/filename/0_real/
                for image_name in os.listdir(os.path.join(file_path, '0_real')):
                    self.data_list.append({
                        "image_path": os.path.join(file_path, '0_real', image_name),
                        "label": 0                   # FIX: was empty label
                    })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path = sample['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception:
            print(f'image error: {image_path}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        # FIX: was `mage` (typo) and `targets` (undefined); label is an int directly
        return image, image_path                     # return path so save_images can mirror structure


# ─────────────────────────────────────────────
# Dataloader helper
# ─────────────────────────────────────────────

def create_dataloader(input_folder, batch_size, shuffle, num_workers, args):
    class PathDataset(Dataset):
        def __init__(self, root):
            self.paths = []
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            for dirpath, _, fnames in os.walk(root):
                if os.path.basename(dirpath) == '0_real':
                    for f in fnames:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                            self.paths.append(os.path.join(dirpath, f))

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert('RGB')
            return self.transform(img), self.paths[idx]

    dataset = PathDataset(input_folder)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# ─────────────────────────────────────────────
# Save images — mirror 0_real → 1_fake
# ─────────────────────────────────────────────

def save_images(images_tensor: torch.Tensor, input_path: str, names: list):
    """
    For every source path  .../0_real/<name>  write the reconstruction to
    .../1_fake/<name>, creating the directory if needed.
    """
    for img_tensor, src_path in zip(images_tensor, names):
        # Replace the '0_real' segment with '1_fake'
        parts = src_path.replace("\\", "/").split("/")
        try:
            idx = parts.index("0_real")
        except ValueError:
            print(f"Warning: '0_real' not found in path {src_path}, skipping.")
            continue

        parts[idx] = "1_fake"
        dst_path = "/".join(parts)                   # FIX: makedirs (not makedir), exist_ok (not exits_ok)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        save_image(img_tensor.cpu(), dst_path)


# ─────────────────────────────────────────────
# Reconstruction
# ─────────────────────────────────────────────

def reconstruct_simple(x, ae, seed, steps=None, tools=None):
    decode_dtype = ae.dtype
    generator = torch.Generator().manual_seed(seed)
    x = x.to(dtype=ae.dtype) * 2.0 - 1.0
    latents = retrieve_latents(ae.encode(x), generator=generator)
    reconstructions = ae.decode(latents.to(decode_dtype), return_dict=False)[0]
    reconstructions = (reconstructions / 2 + 0.5).clamp(0, 1)
    return reconstructions


def recon(pipe, dataloader, ae, seed, args, tools):
    for batch_idx, (images, names) in enumerate(dataloader):
        if batch_idx == 0:
            print(f"Batch {batch_idx + 1}:")
            print(f" - Images shape: {images.shape}")
        with torch.no_grad():
            recons = reconstruct_simple(
                x=images.to(device), ae=ae, seed=seed,
                steps=args.steps, tools=tools
            )
        save_images(images_tensor=recons, input_path=args.input_folder, names=names)


# ─────────────────────────────────────────────
# VAE loader
# ─────────────────────────────────────────────

def get_vae(repo_id, return_full=False):
    if 'ldm' in repo_id:
        pipe = DiffusionPipeline.from_pretrained(
            "CompVis/ldm-text2im-large-256", cache_dir="weights"
        )
        return pipe.vqvae
    elif 'stable-diffusion-3' in repo_id:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16,
            cache_dir='weights'
        )
        return pipe.vae
    else:
        raise ValueError(f"Unsupported repo_id: {repo_id}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()                # FIX: was `parser ;` (syntax error)
    parser.add_argument('--repo_id', type=str, default='stable-diffusion-3',  # FIX: missing comma after default
                        help='Which autoencoder to use')
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed = args.seed
    dataloader = create_dataloader(
        args.input_folder,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        args=args
    )
    ae = get_vae(repo_id=args.repo_id).to(device)
    tools = None
    recon(pipe=None, dataloader=dataloader, ae=ae, seed=seed, args=args, tools=tools)


if __name__ == '__main__':
    main()