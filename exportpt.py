import os
from pathlib import Path

import numpy as np
import torch
from torch import nn


from model import ModelArgs, Transformer



def hf_to_pt(hf_model_path, output_path, dtype=torch.bfloat16):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        print("Please run `pip install transformers` to install it")
        return None

    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path)
    hf_dict = hf_model.state_dict()

    # Convert LlamaConfig to ModelArgs
    config = ModelArgs()
    config.dim = hf_model.config.hidden_size
    config.n_layers = hf_model.config.num_hidden_layers
    config.n_heads = hf_model.config.num_attention_heads
    config.n_kv_heads = hf_model.config.num_attention_heads
    config.vocab_size = hf_model.config.vocab_size
    config.hidden_dim = hf_model.config.intermediate_size
    config.norm_eps = hf_model.config.rms_norm_eps
    config.max_seq_len = hf_model.config.max_position_embeddings

    # Create a new Transformer object and set weights
    model = Transformer(config)

    model.tok_embeddings.weight = nn.Parameter(hf_dict['model.embed_tokens.weight'].to(dtype))
    model.norm.weight = nn.Parameter(hf_dict['model.norm.weight'].to(dtype))

    # Huggingface permutes WQ and WK, this function reverses it
    def permute_reverse(w, n_heads=config.n_heads, dim1=config.dim, dim2=config.dim):
        return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(hf_dict[f'model.layers.{i}.input_layernorm.weight'].to(dtype))
        layer.attention.wq.weight = nn.Parameter(permute_reverse(hf_dict[f'model.layers.{i}.self_attn.q_proj.weight']).to(dtype))
        layer.attention.wk.weight = nn.Parameter(permute_reverse(hf_dict[f'model.layers.{i}.self_attn.k_proj.weight']).to(dtype))
        layer.attention.wv.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.v_proj.weight'].to(dtype))
        layer.attention.wo.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.o_proj.weight'].to(dtype))
        layer.ffn_norm.weight = nn.Parameter(hf_dict[f'model.layers.{i}.post_attention_layernorm.weight'].to(dtype))
        layer.feed_forward.w1.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.gate_proj.weight'].to(dtype))
        layer.feed_forward.w2.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.down_proj.weight'].to(dtype))
        layer.feed_forward.w3.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.up_proj.weight'].to(dtype))

    # Final classifier
    model.output.weight = nn.Parameter(hf_dict['lm_head.weight'].to(dtype))
    model.eval

# Save the model and model_args as a dictionary
    checkpoint = {
    'model': model.state_dict(),
    'model_args': config.__dict__
    }
    torch.save(checkpoint, output_path)

hf_model_path = "D:/oobabooga_windows/text-generation-webui/models/hfl_chinese-llama-2-1.3b"
output_path = "save/converted_model.pt"
hf_to_pt(hf_model_path, output_path)