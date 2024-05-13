import argparse
import logging

import torch
import numpy as np

from model import GPT
from gen_token import generate_token
from transformers import GPT2Tokenizer

def generate_text(input_text, num_tokens_to_generate, model_config, num_runs):
    # tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # encode token
    input_token_ids = tokenizer.encode(input_text, return_tensors="pt")[0].tolist()

    # init model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        n_layer=model_config["n_layer"],
        n_embd=model_config["n_embd"],
        n_head=model_config["n_head"],
        n_positions=model_config["n_positions"],
        attn_pdrop=model_config["attn_pdrop"],
        embd_pdrop=model_config["embd_pdrop"],
        resid_pdrop=model_config["resid_pdrop"],
        layer_norm_epsilon=model_config["layer_norm_epsilon"],
    )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    model.to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings_seq=np.zeros((num_runs,1))
    # Generate tokens
    for r in range(num_runs):
        starter.record()
        for _ in range(num_tokens_to_generate):
            new_token_id = generate_token(
                model,
                input_token_ids,
                temperature=1.0,
                sample=True,
                top_k=None,
                device=device
            )
            input_token_ids.append(new_token_id)
        ender.record()        
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings_seq[r] = curr_time
        
    mean_seq = np.sum(timings_seq) / num_runs
    std_seq = np.std(timings_seq)


    # Decode the generated tokens
    generated_text = tokenizer.decode(input_token_ids)

    return generated_text, mean_seq, std_seq