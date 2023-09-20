import torch
import numpy as np

from scripts.sample_diffusion import load_model
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from einops import rearrange

from ldm.data.scene2scene import Scene2SceneValidation


def ldm_cond_sample(config_path, ckpt_path, dataset, batch_size):
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    x = next(iter(dataloader))

    seg = x['segmentation']

    with torch.no_grad():
        seg = rearrange(seg, 'b h w c -> b c h w')
        condition = model.to_rgb(seg)

        seg = seg.to('cuda').float()
        seg = model.get_learned_conditioning(seg)

        samples, _ = model.sample_log(cond=seg, batch_size=batch_size, ddim=True,
                                      ddim_steps=200, eta=1.)

        samples = model.decode_first_stage(samples)

    save_image(condition, 'cond.png')
    save_image(samples, 'sample.png')


if __name__ == '__main__':

    config_path = '/media/seven/HD_12/code/latent_test/logs/first_model/configs/2023-09-11T05-05-44-project.yaml'
    ckpt_path = '/media/seven/HD_12/code/latent_test/logs/first_model/checkpoints/epoch=000185.ckpt'

    dataset = Scene2SceneValidation(size=256)

    ldm_cond_sample(config_path, ckpt_path, dataset, 1)
