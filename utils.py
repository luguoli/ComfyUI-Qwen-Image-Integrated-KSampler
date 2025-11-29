import torch
import comfy.sample
import comfy.utils
import comfy.model_management
import latent_preview
import node_helpers
import math
import gc
import numpy as np
from PIL import Image
from server import PromptServer
import psutil
import ctypes
from ctypes import wintypes
import time
import platform
import subprocess


def log(message, message_type='info'):
    print(f"[{message_type.upper()}] {message}")

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image:Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def image2mask(image:Image) -> torch.Tensor:
    if image.mode == 'L':
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])
    else:
        image = image.convert('RGB').split()[0]
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])

# 向上取整数倍
def num_round_up_to_multiple(number: int, multiple: int) -> int:
    remainder = number % multiple
    if remainder == 0:
        return number
    else:
        factor = (number + multiple - 1) // multiple  # 向上取整的计算方式
        return factor * multiple

def is_valid_mask(tensor:torch.Tensor) -> bool:
    return not bool(torch.all(tensor == 0).item())

def fit_resize_image(image:Image, target_width:int, target_height:int, fit:str, resize_sampler:str, background_color:str = '#000000') -> Image:
    image = image.convert('RGB')
    orig_width, orig_height = image.size
    if image is not None:
        if fit == 'letterbox':
            if orig_width / orig_height > target_width / target_height:  # 更宽，上下留黑
                fit_width = target_width
                fit_height = int(target_width / orig_width * orig_height)
            else:  # 更瘦，左右留黑
                fit_height = target_height
                fit_width = int(target_height / orig_height * orig_width)
            fit_image = image.resize((fit_width, fit_height), resize_sampler)
            ret_image = Image.new('RGB', size=(target_width, target_height), color=background_color)
            ret_image.paste(fit_image, box=((target_width - fit_width)//2, (target_height - fit_height)//2))
        elif fit == 'crop':
            if orig_width / orig_height > target_width / target_height:  # 更宽，裁左右
                fit_width = int(orig_height * target_width / target_height)
                fit_image = image.crop(
                    ((orig_width - fit_width)//2, 0, (orig_width - fit_width)//2 + fit_width, orig_height))
            else:   # 更瘦，裁上下
                fit_height = int(orig_width * target_height / target_width)
                fit_image = image.crop(
                    (0, (orig_height-fit_height)//2, orig_width, (orig_height-fit_height)//2 + fit_height))
            ret_image = fit_image.resize((target_width, target_height), resize_sampler)
        else:
            ret_image = image.resize((target_width, target_height), resize_sampler)
    return  ret_image


def image_scale_by_aspect_ratio(aspect_ratio, proportional_width, proportional_height,
                                    fit, method, round_to_multiple, scale_to_side, scale_to_length,
                                    background_color,
                                    image=None, mask = None,
                                    ):
        orig_images = []
        orig_masks = []
        orig_width = 0
        orig_height = 0
        target_width = 0
        target_height = 0
        ratio = 1.0
        ret_images = []
        ret_masks = []
        if image is not None:
            for i in image:
                i = torch.unsqueeze(i, 0)
                orig_images.append(i)
            orig_width, orig_height = tensor2pil(orig_images[0]).size
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            for m in mask:
                m = torch.unsqueeze(m, 0)
                if not is_valid_mask(m) and m.shape==torch.Size([1,64,64]):
                    log(f"Warning: input mask is empty, ignore it.", message_type='warning')
                else:
                    orig_masks.append(m)

            if len(orig_masks) > 0:
                _width, _height = tensor2pil(orig_masks[0]).size
                if (orig_width > 0 and orig_width != _width) or (orig_height > 0 and orig_height != _height):
                    log(f"Error: execute failed, because the mask is does'nt match image.", message_type='error')
                    return (None, None, None, 0, 0,)
                elif orig_width + orig_height == 0:
                    orig_width = _width
                    orig_height = _height

        if orig_width + orig_height == 0:
            log(f"Error: execute failed, because the image or mask at least one must be input.", message_type='error')
            return (None, None, None, 0, 0,)

        if aspect_ratio == 'original':
            ratio = orig_width / orig_height
        elif aspect_ratio == 'custom':
            ratio = proportional_width / proportional_height
        else:
            s = aspect_ratio.split(":")
            ratio = int(s[0]) / int(s[1])

        # calculate target width and height
        if scale_to_side == 'max_size':
            if isinstance(scale_to_length, (tuple, list)):
                max_w, max_h = scale_to_length
            elif isinstance(scale_to_length, str) and ',' in scale_to_length:
                max_w, max_h = [int(x) for x in scale_to_length.split(',')]
            else:
                max_w = max_h = int(scale_to_length)
            if orig_width <= max_w and orig_height <= max_h:
                target_width = orig_width
                target_height = orig_height
            elif ratio > max_w / max_h:
                target_width = max_w
                target_height = int(target_width / ratio)
            else:
                target_height = max_h
                target_width = int(target_height * ratio)
        elif ratio > 1:
            if scale_to_side == 'longest':
                target_width = scale_to_length
                target_height = int(target_width / ratio)
            elif scale_to_side == 'shortest':
                target_height = scale_to_length
                target_width = int(target_height * ratio)
            elif scale_to_side == 'width':
                target_width = scale_to_length
                target_height = int(target_width / ratio)
            elif scale_to_side == 'height':
                target_height = scale_to_length
                target_width = int(target_height * ratio)
            elif scale_to_side == 'total_pixel(kilo pixel)':
                target_width = math.sqrt(ratio * scale_to_length * 1000)
                target_height = target_width / ratio
                target_width = int(target_width)
                target_height = int(target_height)
            else:
                target_width = orig_width
                target_height = int(target_width / ratio)
        else:
            if scale_to_side == 'longest':
                target_height = scale_to_length
                target_width = int(target_height * ratio)
            elif scale_to_side == 'shortest':
                target_width = scale_to_length
                target_height = int(target_width / ratio)
            elif scale_to_side == 'width':
                target_width = scale_to_length
                target_height = int(target_width / ratio)
            elif scale_to_side == 'height':
                target_height = scale_to_length
                target_width = int(target_height * ratio)
            elif scale_to_side == 'total_pixel(kilo pixel)':
                target_width = math.sqrt(ratio * scale_to_length * 1000)
                target_height = target_width / ratio
                target_width = int(target_width)
                target_height = int(target_height)
            else:
                target_height = orig_height
                target_width = int(target_height * ratio)

        if round_to_multiple != 'None':
            multiple = int(round_to_multiple)
            target_width = num_round_up_to_multiple(target_width, multiple)
            target_height = num_round_up_to_multiple(target_height, multiple)

        _mask = Image.new('L', size=(target_width, target_height), color='black')
        _image = Image.new('RGB', size=(target_width, target_height), color='black')

        resize_sampler = Image.LANCZOS
        if method == "bicubic":
            resize_sampler = Image.BICUBIC
        elif method == "hamming":
            resize_sampler = Image.HAMMING
        elif method == "bilinear":
            resize_sampler = Image.BILINEAR
        elif method == "box":
            resize_sampler = Image.BOX
        elif method == "nearest":
            resize_sampler = Image.NEAREST

        if len(orig_images) > 0:
            for i in orig_images:
                _image = tensor2pil(i).convert('RGB')
                _image = fit_resize_image(_image, target_width, target_height, fit, resize_sampler, background_color)
                ret_images.append(pil2tensor(_image))
        if len(orig_masks) > 0:
            for m in orig_masks:
                _mask = tensor2pil(m).convert('L')
                _mask = fit_resize_image(_mask, target_width, target_height, fit, resize_sampler).convert('L')
                ret_masks.append(image2mask(_mask))
        if len(ret_images) > 0 and len(ret_masks) >0:
            log(f"Processed {len(ret_images)} image(s).", message_type='finish')
            return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),[orig_width, orig_height], target_width, target_height,)
        elif len(ret_images) > 0 and len(ret_masks) == 0:
            log(f"Processed {len(ret_images)} image(s).", message_type='finish')
            return (torch.cat(ret_images, dim=0), None, [orig_width, orig_height], target_width, target_height,)
        elif len(ret_images) == 0 and len(ret_masks) > 0:
            log(f"Processed {len(ret_masks)} image(s).", message_type='finish')
            return (None, torch.cat(ret_masks, dim=0), [orig_width, orig_height], target_width, target_height,)
        else:
            log(f"Error: skipped, because the available image or mask is not found.", message_type='error')
            return (None, None, None, 0, 0,)



def cleanGPUUsedForce():
    gc.collect()
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    PromptServer.instance.prompt_queue.set_flag("free_memory", True)

def get_system_prompt(instruction):
    template_prefix = "<|im_start|>system\n"
    template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    instruction_content = ""
    if instruction == "":
        instruction_content = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
    else:
        # for handling mis use of instruction
        if template_prefix in instruction:
            # remove prefix from instruction
            instruction = instruction.split(template_prefix)[1]
        if template_suffix in instruction:
            # remove suffix from instruction
            instruction = instruction.split(template_suffix)[0]
        if "{}" in instruction:
            # remove {} from instruction
            instruction = instruction.replace("{}", "")
        instruction_content = instruction
    llama_template = template_prefix + instruction_content + template_suffix

    return llama_template

def clean_ram(clean_file_cache=True, clean_processes=True, clean_dlls=True, retry_times=3, anything=None, unique_id=None, extra_pnginfo=None):
    try:
        def get_ram_usage():
            memory = psutil.virtual_memory()
            return memory.percent, memory.available / (1024 * 1024)

        before_usage, before_available = get_ram_usage()
        system = platform.system()

        for attempt in range(retry_times):
            if clean_file_cache:
                try:
                    if system == "Windows":
                        ctypes.windll.kernel32.SetSystemFileCacheSize(-1, -1, 0)
                    elif system == "Linux":
                        subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                                      check=False, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                except:
                    pass

            if clean_processes:
                if system == "Windows":
                    for process in psutil.process_iter(['pid', 'name']):
                        try:
                            handle = ctypes.windll.kernel32.OpenProcess(
                                wintypes.DWORD(0x001F0FFF),
                                wintypes.BOOL(False),
                                wintypes.DWORD(process.info['pid'])
                            )
                            ctypes.windll.psapi.EmptyWorkingSet(handle)
                            ctypes.windll.kernel32.CloseHandle(handle)
                        except:
                            continue

            if clean_dlls:
                try:
                    if system == "Windows":
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                    elif system == "Linux":
                        subprocess.run(["sync"], check=True)
                except:
                    pass

            time.sleep(1)

        after_usage, after_available = get_ram_usage()
        freed_mb = after_available - before_available
        print(f"RAM清理完成 / RAM cleanup completed [{before_usage:.1f}% → {after_usage:.1f}%, 释放 / Freed: {freed_mb:.0f}MB]")

    except Exception as e:
        print(f"RAM清理失败 / RAM cleanup failed: {str(e)}")

    return anything



# ----------------优化官方扩展的TextEncodeQwenImageEditPlus，解决生成图片被偏移、放大、缩小的问题，保持官方扩展的遵从指令的优点----------------
def get_image_prompt(vae=None, image1=None, image2=None, image3=None, image4=None, image5=None,upscale_method="lanczos",crop="disabled",instruction=""):
    ref_latents = []
    images = [image1, image2, image3, image4, image5]
    images_vl = []
    llama_template = get_system_prompt(instruction)
    image_prompt = ""

    for i, image in enumerate(images):
        if image is not None:
            samples = image.movedim(-1, 1)

            # -----------------------------
            # 1. Vision-Language path (384x384)
            # -----------------------------
            total = int(384 * 384)

            scale_by = (total / (samples.shape[2] * samples.shape[3])) ** 0.5
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)

            s_vl = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
            images_vl.append(s_vl.movedim(1, -1))

            # -----------------------------
            # 2. VAE latent path (strict original resolution)
            # -----------------------------
            if vae is not None:
                # 使用原图宽高，不放大
                width = (samples.shape[3] + 7) // 8 * 8
                height = (samples.shape[2] + 7) // 8 * 8
                s_vae = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

            image_prompt += f"Picture {i+1}: <|vision_start|><|image_pad|><|vision_end|>"

    return (image_prompt, images_vl, llama_template, ref_latents)
def prompt_encode(clip, prompt, image_prompt="", images_vl=[], llama_template=None, ref_latents=[]):
    if len(images_vl) > 0:
        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
    else:
        tokens = clip.tokenize(prompt)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    if len(ref_latents) > 0:
        conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
    return conditioning
# ----------------优化官方扩展的TextEncodeQwenImageEditPlus，解决生成图片被偏移、放大、缩小的问题，保持官方扩展的遵从指令的优点----------------


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):

    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    
    # print("✅ 采样完成/Sampling Complete!")
    
    out = latent.copy()
    out["samples"] = samples

    # print("🎉 Easy KSampler 执行完成!\n")

    return out

