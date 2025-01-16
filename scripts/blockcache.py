##  First Block Cache / TeaCache for Forge webui
##  with option to skip cache for early steps
##  always processes last step
##  handles highresfix
##  handles PAG and SAG (with unet models, not Flux) by accelerating them too, independently
##      opposite time/quality trade offs ... but some way of handling them is necessary to avoid potential errors

##  derived from https://github.com/likelovewant/sd-forge-teacache

import torch
import numpy as np
from torch import Tensor
import gradio as gr
from modules import scripts
from modules.ui_components import InputAccordion
from backend.nn.flux import IntegratedFluxTransformer2DModel
from backend.nn.flux import timestep_embedding as timestep_embedding_flux
from backend.nn.unet import IntegratedUNet2DConditionModel, apply_control
from backend.nn.unet import timestep_embedding as timestep_embedding_unet

class BlockCache(scripts.Script):
    original_inner_forward = None
    
    def __init__(self):
        if BlockCache.original_inner_forward is None:
            BlockCache.original_inner_forward = IntegratedFluxTransformer2DModel.inner_forward
            BlockCache.original_forward_unet = IntegratedUNet2DConditionModel.forward

    def title(self):
        return "First Block Cache / TeaCache"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enabled:
            method = gr.Radio(label="Method", choices=["First Block Cache", "TeaCache"], type="value", value="First Block Cache")
            with gr.Row():
                nocache_steps = gr.Number(label="Uncached starting steps", scale=0,
                    minimum=1, maximum=12, value=1, step=1,
                )
                threshold = gr.Slider(label="caching threshold, higher values cache more aggressively.", 
                    minimum=0.0, maximum=1.0, value=0.1, step=0.01,
                )

        enabled.do_not_save_to_config = True
        method.do_not_save_to_config = True
        nocache_steps.do_not_save_to_config = True
        threshold.do_not_save_to_config = True

        self.infotext_fields = [
            (enabled, lambda d: d.get("bc_enabled", False)),
            (method,        "bc_method"),
            (threshold,     "bc_threshold"),
            (nocache_steps, "bc_nocache_steps"),
        ]

        return [enabled, method, threshold, nocache_steps]


    def process(self, p, *args):
        enabled, method, threshold, nocache_steps = args

        if enabled:
            if method == "First Block Cache":
                if (p.sd_model.is_sd1 == True) or (p.sd_model.is_sd2 == True) or (p.sd_model.is_sdxl == True):
                    IntegratedUNet2DConditionModel.forward = patched_forward_unet_fbc
                else:
                    IntegratedFluxTransformer2DModel.inner_forward = patched_inner_forward_flux_fbc
            else:
                if (p.sd_model.is_sd1 == True) or (p.sd_model.is_sd2 == True) or (p.sd_model.is_sdxl == True):
                    IntegratedUNet2DConditionModel.forward = patched_forward_unet_tc
                else:
                    IntegratedFluxTransformer2DModel.inner_forward = patched_inner_forward_flux_tc

            p.extra_generation_params.update({
                "bc_enabled"        : enabled,
                "bc_method"         : method,
                "bc_threshold"      : threshold,
                "bc_nocache_steps"  : nocache_steps,
            })

            setattr(BlockCache, "threshold", threshold)
            setattr(BlockCache, "nocache_steps", nocache_steps)


    def process_before_every_sampling(self, p, *args, **kwargs):
        enabled = args[0]

        #   possibly four passes through the forward method on each step: cond, uncond, PAG, SAG

        if enabled:
            setattr(BlockCache, "index", 0)
            setattr(BlockCache, "distance", [0])
            setattr(BlockCache, "this_step", 0)
            setattr(BlockCache, "last_step", p.hr_second_pass_steps if p.is_hr_pass else p.steps)
            setattr(BlockCache, "residual", [None])
            setattr(BlockCache, "previous", [None])
            setattr(BlockCache, "previousSigma", None)

    def postprocess(self, params, processed, *args):
        # always clean up after processing
        enabled = args[0]
        if enabled:
            # restore the original inner_forward method
            IntegratedFluxTransformer2DModel.inner_forward = BlockCache.original_inner_forward
            IntegratedUNet2DConditionModel.forward = BlockCache.original_forward_unet

            delattr(BlockCache, "index")
            delattr(BlockCache, "threshold")
            delattr(BlockCache, "nocache_steps")
            delattr(BlockCache, "distance")
            delattr(BlockCache, "this_step")
            delattr(BlockCache, "last_step")
            delattr(BlockCache, "residual")
            delattr(BlockCache, "previous")
            delattr(BlockCache, "previousSigma")


def patched_inner_forward_flux_fbc(self, img, img_ids, txt, txt_ids, timesteps, y, guidance=None):
    # BlockCache version
    
    thisSigma = timesteps.item()
    if BlockCache.previousSigma == thisSigma:
        BlockCache.index += 1
        if BlockCache.index == len(BlockCache.distance):
            BlockCache.distance.append(0)
            BlockCache.residual.append(None)
            BlockCache.previous.append(None)
    else:
        BlockCache.previousSigma = thisSigma
        BlockCache.index = 0
        BlockCache.this_step += 1

    index    = BlockCache.index

    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # Image and text embedding
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding_flux(timesteps, 256).to(img.dtype))

    # If guidance_embed is enabled, add guidance information
    if self.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(timestep_embedding_flux(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y)
    txt = self.txt_in(txt)

    # Merge image and text IDs
    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    original_img = img.clone()

    first_block = True
    for block in self.double_blocks:
        img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
        if first_block:
            first_block = False

            if BlockCache.this_step > BlockCache.nocache_steps and BlockCache.this_step != BlockCache.last_step:
                BlockCache.distance[index] += ((img - BlockCache.previous[index]).abs().mean() / BlockCache.previous[index].abs().mean()).cpu().item()
                BlockCache.previous[index] = img.clone() ##clone?, seemed OK without
                if BlockCache.distance[index] < BlockCache.threshold:
                    img = original_img + BlockCache.residual[index]
                    img = self.final_layer(img, vec)
                    return img      ##  early exit
            else:
                BlockCache.previous[index] = img

    img = torch.cat((txt, img), 1)
    for block in self.single_blocks:
        img = block(img, vec=vec, pe=pe)
    img = img[:, txt.shape[1]:, ...]
    BlockCache.residual[index] = img - original_img
    BlockCache.distance[index] = 0

    img = self.final_layer(img, vec)
    return img

def patched_inner_forward_flux_tc(self, img, img_ids, txt, txt_ids, timesteps, y, guidance=None):
    # TeaCache version

    thisSigma = timesteps.item()
    if BlockCache.previousSigma == thisSigma:
        BlockCache.index += 1
        if BlockCache.index == len(BlockCache.distance):
            BlockCache.distance.append(0)
            BlockCache.residual.append(None)
            BlockCache.previous.append(None)
    else:
        BlockCache.previousSigma = thisSigma
        BlockCache.index = 0
        BlockCache.this_step += 1

    index    = BlockCache.index

    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # Image and text embedding
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding_flux(timesteps, 256).to(img.dtype))

    # If guidance_embed is enabled, add guidance information
    if self.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(timestep_embedding_flux(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y)
    txt = self.txt_in(txt)

    # Merge image and text IDs
    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    original_img = img.clone()

    if BlockCache.this_step <= BlockCache.nocache_steps:
        should_calc = True
    elif BlockCache.this_step == BlockCache.last_step:
        should_calc = True
    elif BlockCache.previous[index] is None or BlockCache.previous[index].shape != original_img.shape:
        # should be redundant check if previous exists and has the correct shape
        should_calc = True
    else:
        coefficients = [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
        rescale_func = np.poly1d(coefficients)
        BlockCache.distance[index] += rescale_func(
            ((original_img - BlockCache.previous[index]).abs().mean() / BlockCache.previous[index].abs().mean()).cpu().item()
        )

        if BlockCache.distance[index] < BlockCache.threshold:
            should_calc = False
        else:
            should_calc = True

    BlockCache.previous[index] = original_img

    if should_calc or BlockCache.residual[index] is None:
        BlockCache.distance[index] = 0
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1]:, ...]
        BlockCache.residual[index] = img - original_img
    else:
        img += BlockCache.residual[index]

    img = self.final_layer(img, vec)
    return img

def patched_forward_unet_fbc(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
    # BlockCache version

    thisSigma = transformer_options["sigmas"].item()
    if BlockCache.previousSigma == thisSigma:
        BlockCache.index += 1
        if BlockCache.index == len(BlockCache.distance):
            BlockCache.distance.append(0)
            BlockCache.residual.append(None)
            BlockCache.previous.append(None)
    else:
        BlockCache.previousSigma = thisSigma
        BlockCache.index = 0
        BlockCache.this_step += 1

    index    = BlockCache.index
    residual = BlockCache.residual[index]
    previous = BlockCache.previous[index]
    distance = BlockCache.distance[index]

    transformer_options["original_shape"] = list(x.shape)
    transformer_options["transformer_index"] = 0
    transformer_patches = transformer_options.get("patches", {})
    block_modifiers = transformer_options.get("block_modifiers", [])
    assert (y is not None) == (self.num_classes is not None)
    hs = []
    t_emb = timestep_embedding_unet(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
    emb = self.time_embed(t_emb)
    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)
    h = x

    original_h = h.clone()

    skip = False
    first_block = True
    for id, module in enumerate(self.input_blocks):
        transformer_options["block"] = ("input", id)
        for block_modifier in block_modifiers:
            h = block_modifier(h, 'before', transformer_options)
        h = module(h, emb, context, transformer_options)
        h = apply_control(h, control, 'input')
        for block_modifier in block_modifiers:
            h = block_modifier(h, 'after', transformer_options)
        if "input_block_patch" in transformer_patches:
            patch = transformer_patches["input_block_patch"]
            for p in patch:
                h = p(h, transformer_options)
        hs.append(h)
        if "input_block_patch_after_skip" in transformer_patches:
            patch = transformer_patches["input_block_patch_after_skip"]
            for p in patch:
                h = p(h, transformer_options)

        if first_block:
            first_block = False
            if BlockCache.this_step > BlockCache.nocache_steps and BlockCache.this_step < BlockCache.last_step and previous is not None:
                distance += ((h - previous).abs().mean() / previous.abs().mean()).cpu().item()
                previous = h.clone()
                if distance < BlockCache.threshold:
                    h = original_h + residual
                    skip = True
                    break
            else:
                previous = h.clone()

    if not skip:
        transformer_options["block"] = ("middle", 0)
        for block_modifier in block_modifiers:
            h = block_modifier(h, 'before', transformer_options)
        h = self.middle_block(h, emb, context, transformer_options)
        h = apply_control(h, control, 'middle')
        for block_modifier in block_modifiers:
            h = block_modifier(h, 'after', transformer_options)
        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control(hsp, control, 'output')
            if "output_block_patch" in transformer_patches:
                patch = transformer_patches["output_block_patch"]
                for p in patch:
                    h, hsp = p(h, hsp, transformer_options)
            h = torch.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                output_shape = hs[-1].shape
            else:
                output_shape = None
            for block_modifier in block_modifiers:
                h = block_modifier(h, 'before', transformer_options)
            h = module(h, emb, context, transformer_options, output_shape)
            for block_modifier in block_modifiers:
                h = block_modifier(h, 'after', transformer_options)
        transformer_options["block"] = ("last", 0)
        for block_modifier in block_modifiers:
            h = block_modifier(h, 'before', transformer_options)
        if "group_norm_wrapper" in transformer_options:
            out_norm, out_rest = self.out[0], self.out[1:]
            h = transformer_options["group_norm_wrapper"](out_norm, h, transformer_options)
            h = out_rest(h)
        else:
            h = self.out(h)
        for block_modifier in block_modifiers:
            h = block_modifier(h, 'after', transformer_options)

        residual = h - original_h
        distance = 0

    BlockCache.residual[index] = residual
    BlockCache.previous[index] = previous
    BlockCache.distance[index] = distance

    return h.type(x.dtype)


def patched_forward_unet_tc(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
    # TeaCache version

    thisSigma = transformer_options["sigmas"].item()
    if BlockCache.previousSigma == thisSigma:
        BlockCache.index += 1
        if BlockCache.index == len(BlockCache.distance):
            BlockCache.distance.append(0)
            BlockCache.residual.append(None)
            BlockCache.previous.append(None)
    else:
        BlockCache.previousSigma = thisSigma
        BlockCache.index = 0
        BlockCache.this_step += 1

    index    = BlockCache.index
    residual = BlockCache.residual[index]
    previous = BlockCache.previous[index]
    distance = BlockCache.distance[index]

    transformer_options["original_shape"] = list(x.shape)
    transformer_options["transformer_index"] = 0
    transformer_patches = transformer_options.get("patches", {})
    block_modifiers = transformer_options.get("block_modifiers", [])
    assert (y is not None) == (self.num_classes is not None)
    hs = []
    t_emb = timestep_embedding_unet(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
    emb = self.time_embed(t_emb)
    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)
    h = x

    original_h = h.clone()

    if BlockCache.this_step <= BlockCache.nocache_steps:
        should_calc = True
    elif BlockCache.this_step == BlockCache.last_step:
        should_calc = True
    elif previous is None or previous.shape != original_h.shape:
        # should be redundant check if previous exists and has the correct shape
        should_calc = True
    else:
        distance += ((original_h - previous).abs().mean() / previous.abs().mean()).cpu().item()

        if distance < BlockCache.threshold:
            should_calc = False
        else:
            should_calc = True

    if should_calc:
        for id, module in enumerate(self.input_blocks):
            transformer_options["block"] = ("input", id)
            for block_modifier in block_modifiers:
                h = block_modifier(h, 'before', transformer_options)
            h = module(h, emb, context, transformer_options)
            h = apply_control(h, control, 'input')
            for block_modifier in block_modifiers:
                h = block_modifier(h, 'after', transformer_options)
            if "input_block_patch" in transformer_patches:
                patch = transformer_patches["input_block_patch"]
                for p in patch:
                    h = p(h, transformer_options)
            hs.append(h)
            if "input_block_patch_after_skip" in transformer_patches:
                patch = transformer_patches["input_block_patch_after_skip"]
                for p in patch:
                    h = p(h, transformer_options)

        transformer_options["block"] = ("middle", 0)
        for block_modifier in block_modifiers:
            h = block_modifier(h, 'before', transformer_options)
        h = self.middle_block(h, emb, context, transformer_options)
        h = apply_control(h, control, 'middle')
        for block_modifier in block_modifiers:
            h = block_modifier(h, 'after', transformer_options)
        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control(hsp, control, 'output')
            if "output_block_patch" in transformer_patches:
                patch = transformer_patches["output_block_patch"]
                for p in patch:
                    h, hsp = p(h, hsp, transformer_options)
            h = torch.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                output_shape = hs[-1].shape
            else:
                output_shape = None
            for block_modifier in block_modifiers:
                h = block_modifier(h, 'before', transformer_options)
            h = module(h, emb, context, transformer_options, output_shape)
            for block_modifier in block_modifiers:
                h = block_modifier(h, 'after', transformer_options)
        transformer_options["block"] = ("last", 0)
        for block_modifier in block_modifiers:
            h = block_modifier(h, 'before', transformer_options)
        if "group_norm_wrapper" in transformer_options:
            out_norm, out_rest = self.out[0], self.out[1:]
            h = transformer_options["group_norm_wrapper"](out_norm, h, transformer_options)
            h = out_rest(h)
        else:
            h = self.out(h)
        for block_modifier in block_modifiers:
            h = block_modifier(h, 'after', transformer_options)

        residual = h - original_h
        distance = 0
    else:
        h += residual

    BlockCache.residual[index] = residual
    BlockCache.previous[index] = original_h
    BlockCache.distance[index] = distance

    return h.type(x.dtype)

