import functools
import math
import torch.nn
from modules import shared, scripts, processing, script_callbacks
from lib_scalecrafter import global_state


class Script(scripts.Script):
    def title(self):
        return "ScaleCrafter"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p: processing.StableDiffusionProcessing, *args):
        global_state.enable = True
        train_size = 1024 if p.sd_model.is_sdxl else 512
        global_state.dilation = math.sqrt(p.width * p.height) / train_size
        global_state.outer_blocks = 4
        global_state.stop_at_step = 10

    def postprocess_batch(self, p, *args, **kwargs):
        global_state.current_sampling_step = 0


def increment_sampling_step(*args, **kwargs):
    global_state.current_sampling_step += 1


script_callbacks.on_cfg_denoised(increment_sampling_step)


def make_dilate_model(model):
    unet = model.model.diffusion_model

    for in_block_i, in_block in enumerate(reversed(unet.input_blocks)):
        patch_conv2d(in_block, "down", in_block_i)

    for mid_block in unet.middle_block:
        patch_conv2d(mid_block, "middle", 0)

    for out_block_i, out_block in enumerate(unet.output_blocks):
        patch_conv2d(out_block, "up", out_block_i)


script_callbacks.on_model_loaded(make_dilate_model)


def patch_conv2d(module, block_type, block_index):
    for name, submodule in module.named_modules():
        if isinstance(submodule, torch.nn.Conv2d) and (submodule.kernel_size == 3 or submodule.kernel_size == (3, 3)):
            submodule.forward = functools.partial(
                conv2d_forward_patch,
                block_type=block_type,
                block_index=block_index,
                self=submodule,
                original_function=submodule.forward,
            )


def conv2d_forward_patch(x, *args, block_type, block_index, self, original_function, **kwargs):
    if not (
        global_state.enable and
        global_state.current_sampling_step < global_state.stop_at_step and (
            block_type in {"down", "up"} and block_index < global_state.outer_blocks or
            block_type == "middle"
        )
    ):
        return original_function(x, *args, **kwargs)

    dilation_ceil = math.ceil(global_state.dilation)

    scale_factor = dilation_ceil / global_state.dilation
    original_size = x.shape[-2:]
    x = torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode="bilinear")

    original_dilation = self.dilation
    original_padding = self.padding
    try:
        self.dilation = (dilation_ceil, dilation_ceil)
        self.padding = (dilation_ceil, dilation_ceil)
        x_out = original_function(x, *args, **kwargs)
    finally:
        self.dilation = original_dilation
        self.padding = original_padding

    x_out = torch.nn.functional.interpolate(x_out, size=original_size, mode="bilinear")
    return x_out
