import torch
from diffusers import AutoencoderKL, DDPMScheduler
from models import UNet2DConditionModel
from transformers import CLIPVisionModel, CLIPFeatureExtractor
from image_to_image_pipeline_cfg import StableDiffusionImage2ImagePipeline
import os
from PIL import Image
import argparse
import torch.nn.functional as F
from torchvision import transforms as T

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--num_tokens", type=int, required=True
    )
    args = parser.parse_args()
    return args

args = parse_args()

outpath = f"./reconstruction"
os.makedirs(os.path.join(outpath), exist_ok=True)

if args.num_tokens == 256:
    use_pos_embeds = True
else:
    use_pos_embeds = False

unet = UNet2DConditionModel.from_pretrained(
    "./ckpts/stable-diffusion-xl-base-1.0", subfolder=f"unet_{str(args.num_tokens)}" ,revision=None, low_cpu_mem_usage=False, device_map=None,
    mapping_dim=1024, mapping_dim1=2048, num_tokens=args.num_tokens, use_embeds=use_pos_embeds
)
unet = unet.to(dtype=torch.float32)

image_encoder = CLIPVisionModel.from_pretrained("./ckpts/clip-vit-large-patch14").to(dtype=torch.float32)
vae = AutoencoderKL.from_pretrained(
    "./ckpts/stable-diffusion-xl-base-1.0", subfolder="vae", revision=None, variant=None
).to(dtype=torch.float32)
image_encoder.requires_grad_(False)
image_encoder = image_encoder.to("cuda", dtype=torch.float32)

scheduler = DDPMScheduler.from_pretrained("./ckpts/stable-diffusion-xl-base-1.0", subfolder="scheduler")
feature_extractor = CLIPFeatureExtractor.from_pretrained("./ckpts/clip-vit-large-patch14")

cfg_pipeline = StableDiffusionImage2ImagePipeline(
    vae=vae.to("cuda", dtype=torch.float32),
    unet=unet.to("cuda", dtype=torch.float32),
    scheduler=scheduler,
    feature_extractor=feature_extractor,
    image_encoder=image_encoder.to("cuda", dtype=torch.float32),
)
cfg_pipeline = cfg_pipeline.to("cuda", dtype=torch.float32)
cfg_pipeline.set_progress_bar_config(disable=True)

transform_input = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
])

input_img = "./examples/cat.jpg"
output_img = f"cat_rec_{str(args.num_tokens)}.jpg"
null_img = Image.new("RGB", (224, 224), (0, 0, 0))
null_tensor = transform_input(null_img).to("cuda", dtype=torch.float32)

with torch.no_grad():
    input_img = Image.open(input_img)
    input_img = transform_input(input_img)

    image_embeds = image_encoder(input_img.unsqueeze(0).to("cuda")).last_hidden_state[:, 1:]
    b, n, c = image_embeds.shape
    sqrt_n = int(n**0.5)
    stride = int(sqrt_n // (args.num_tokens ** 0.5))

    image_embeds = image_embeds.permute(0, 2, 1).view(b, c, sqrt_n, sqrt_n)
    image_embeds_cur = F.avg_pool2d(image_embeds, kernel_size=(stride, stride), stride=stride)
    image_embeds_cur = image_embeds_cur.view(b, c, -1).permute(0, 2, 1).contiguous()

    null_image_embeds = image_encoder(null_tensor.unsqueeze(0).to("cuda")).last_hidden_state[:, 1:]
    null_image_embeds = null_image_embeds.permute(0, 2, 1).view(b, c, sqrt_n, sqrt_n)
    null_image_embeds_cur = F.avg_pool2d(null_image_embeds, kernel_size=(stride, stride), stride=stride)
    null_image_embeds_cur = null_image_embeds_cur.view(b, c, -1).permute(0, 2, 1).contiguous()

    cfg_images = cfg_pipeline(input_image_embed=image_embeds_cur, num_inference_steps=50, guidance_scale=2.0, null_input_image_embed=null_image_embeds_cur).images[0]
    cfg_images.save(os.path.join(outpath, output_img))


                

