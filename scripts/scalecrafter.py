import functools
import math
import sys
import textwrap
from typing import List, Tuple
import torch.nn
import gradio as gr
from modules import scripts, processing, script_callbacks, prompt_parser, sd_samplers
from lib_scalecrafter import global_state, hijacker


class Script(scripts.Script):
    def title(self):
        return "ScaleCrafter"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(label=self.title(), open=False):
            enable = gr.Checkbox(
                label="Enabled",
                value=False,
                elem_id=self.elem_id("enable"),
            )

            stop_step_ratio = gr.Slider(
                label="Stop At Step",
                value=0.5,
                minimum=0,
                maximum=1,
                elem_id=self.elem_id("stop_at_step"),
            )

            inner_blocks = gr.Slider(
                label="Inner Blocks",
                value=5,
                minimum=0,
                maximum=12,
                step=1,
                elem_id=self.elem_id("inner_blocks"),
            )

            with gr.Accordion(label="Noise-Damped CFG Scale"):
                enable_damped_cfg = gr.Checkbox(
                    label="Enable Noise-Damped CFG Scale",
                    value=True,
                    elem_id=self.elem_id("noise_damped_enable")
                )
                with gr.Row():
                    damped_blocks_start = gr.Slider(
                        label="First Block (Included)",
                        value=0,
                        minimum=0,
                        maximum=5,
                        step=1,
                        elem_id=self.elem_id("noise_damped_block_start"),
                    )

                    damped_blocks_stop = gr.Slider(
                        label="Last Block (Excluded)",
                        value=5,
                        minimum=0,
                        maximum=12,
                        step=1,
                        elem_id=self.elem_id("noise_damped_block_stop"),
                    )

        damped_blocks_start.release(
            fn=lambda damped_blocks_start, damped_blocks_stop: gr.Slider.update(minimum=damped_blocks_start, value=max(damped_blocks_start, damped_blocks_stop)),
            inputs=[damped_blocks_start, damped_blocks_stop],
            outputs=[damped_blocks_stop],
        )

        damped_blocks_stop.release(
            fn=lambda damped_blocks_stop, damped_blocks_start: gr.Slider.update(maximum=damped_blocks_stop, value=min(damped_blocks_stop, damped_blocks_start)),
            inputs=[damped_blocks_stop, damped_blocks_start],
            outputs=[damped_blocks_start],
        )

        return enable, stop_step_ratio, inner_blocks, enable_damped_cfg, damped_blocks_start, damped_blocks_stop

    def process(self, p: processing.StableDiffusionProcessing, enable, stop_step_ratio, inner_blocks, enable_damped_cfg, damped_blocks_start, damped_blocks_stop, *args, **kwargs):
        global_state.enable = enable

        train_size = 1024 if p.sd_model.is_sdxl else 512
        global_state.dilation = math.sqrt(p.width * p.height) / train_size
        global_state.dilation_x = p.width / train_size
        global_state.dilation_y = p.height / train_size
        global_state.inner_blocks = inner_blocks
        global_state.enable_damped_cfg = enable_damped_cfg
        global_state.damped_blocks_start = damped_blocks_start
        global_state.damped_blocks_stop = damped_blocks_stop
        global_state.stop_step_ratio = stop_step_ratio

    def before_process_batch(self, p, *args, **kwargs):
        global_state.current_step = 0
        global_state.total_steps = p.steps
        global_state.batch_size = p.batch_size

    def before_hr(self, p: processing.StableDiffusionProcessingTxt2Img, *args):
        global_state.total_steps = p.steps

        train_size = 1024 if p.sd_model.is_sdxl else 512
        global_state.dilation = math.sqrt(p.width * p.height) / train_size
        global_state.dilation_x = p.width / train_size
        global_state.dilation_y = p.height / train_size


prompt_parser_hijacker = hijacker.ModuleHijacker.install_or_get(
    module=prompt_parser,
    hijacker_attribute="__scalecrafter_hijacker",
    on_uninstall=script_callbacks.on_script_unloaded,
)


@prompt_parser_hijacker.hijack("get_multicond_learned_conditioning")
def get_multicond_learned_conditioning_hijack(model, prompts, steps, *args, original_function, **kwargs):
    conds = original_function(model, prompts, steps, *args, **kwargs)
    if not global_state.enable:
        return conds

    empty_schedule = prompt_parser.get_learned_conditioning(model, [""], steps, *args, **kwargs)[0]
    conds.batch[0][:0] = [prompt_parser.ComposableScheduledPromptConditioning(schedules=empty_schedule)] * len(conds.batch)
    return conds


def on_cfg_denoiser(params: script_callbacks.CFGDenoiserParams, *args, **kwargs):
    if global_state.enable:
        batch_size = params.text_uncond.shape[0]
        params.text_cond[:batch_size] = params.text_uncond


script_callbacks.on_cfg_denoiser(on_cfg_denoiser)


def on_cfg_denoised(params: script_callbacks.CFGDenoisedParams, *args, **kwargs):
    global_state.current_step += 1


script_callbacks.on_cfg_denoised(on_cfg_denoised)


def combine_denoised_hijack(
    x_out: torch.Tensor,
    batch_cond_indices: List[List[Tuple[int, float]]],
    text_uncond: torch.Tensor,
    cond_scale: float,
    original_function,
) -> torch.Tensor:
    if not global_state.enable:
        return original_function(x_out, batch_cond_indices, text_uncond, cond_scale)

    batch_size = text_uncond.shape[0]
    damped_uncond = x_out[:batch_size]
    denoised = x_out[-batch_size:].clone()

    for i, cond_indices in enumerate(batch_cond_indices):
        if i == 0:
            cond_indices = cond_indices[batch_size:]

        for cond_index, weight in cond_indices:
            denoised[i] += (x_out[cond_index] - damped_uncond[i]) * (weight * cond_scale)

    return denoised


sd_samplers_hijacker = hijacker.ModuleHijacker.install_or_get(
    module=sd_samplers,
    hijacker_attribute='__neutral_prompt_hijacker',
    on_uninstall=script_callbacks.on_script_unloaded,
)


@sd_samplers_hijacker.hijack('create_sampler')
def create_sampler_hijack(name: str, model, original_function):
    sampler = original_function(name, model)
    if not hasattr(sampler, 'model_wrap_cfg') or not hasattr(sampler.model_wrap_cfg, 'combine_denoised'):
        if global_state.enable:
            warn_unsupported_sampler()

        return sampler

    sampler.model_wrap_cfg.combine_denoised = functools.partial(
        combine_denoised_hijack,
        original_function=sampler.model_wrap_cfg.combine_denoised
    )
    return sampler


def warn_unsupported_sampler():
    console_warn('''
        Neutral prompt relies on composition via AND, which the webui does not support when using any of the DDIM, PLMS and UniPC samplers
        The sampler will NOT be patched
        Falling back on original sampler implementation...
    ''')


def warn_projection_not_found():
    console_warn('''
        Could not find a projection for one or more AND_PERP prompts
        These prompts will NOT be made perpendicular
    ''')


def console_warn(message):
    if not global_state.verbose:
        return

    print(f'\n[sd-webui-neutral-prompt extension]{textwrap.dedent(message)}', file=sys.stderr)


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
            if getattr(submodule, "__scalecrafter_hijacked", False):
                continue
            else:
                setattr(submodule, "__scalecrafter_hijacked", True)

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
        global_state.current_step / global_state.total_steps < global_state.stop_step_ratio
    ):
        return original_function(x, *args, **kwargs)

    x_uncond = original_function(x[-global_state.batch_size:], *args, **kwargs)
    out_shape = x_uncond.shape[-2:]
    x_damped_uncond = x[:global_state.batch_size]
    x = x[global_state.batch_size:-global_state.batch_size]

    if (
        block_type in {"down", "up"} and block_index < global_state.inner_blocks or
        block_type == "middle"
    ):
        x = dilate_conv2d_undilate(self, x, global_state.dilation, out_shape, original_function, *args, **kwargs)
    else:
        x = original_function(x, *args, **kwargs)

    if (
        global_state.enable_damped_cfg and
        block_type in {"down", "up"} and global_state.damped_blocks_start <= block_index < global_state.damped_blocks_stop
    ):
        x_damped_uncond = dilate_conv2d_undilate(self, x_damped_uncond, global_state.dilation, out_shape, original_function, *args, **kwargs)
    else:
        x_damped_uncond = original_function(x_damped_uncond, *args, **kwargs)

    return torch.cat([x_damped_uncond, x, x_uncond])


def dilate_conv2d_undilate(self, x, dilation, out_shape, original_function, *args, **kwargs):
    dilation_ceil = math.ceil(dilation)
    x_scale_factor = dilation_ceil / dilation
    x = torch.nn.functional.interpolate(x, scale_factor=x_scale_factor, mode="bilinear")

    original_dilation = self.dilation
    original_padding = self.padding
    try:
        self.dilation = (dilation_ceil, dilation_ceil)
        self.padding = (dilation_ceil, dilation_ceil)
        x = original_function(x, *args, **kwargs)
    finally:
        self.dilation = original_dilation
        self.padding = original_padding

    return torch.nn.functional.interpolate(x, size=out_shape, mode="bilinear")
