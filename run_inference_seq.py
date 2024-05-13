import torch
import csv
from inference import generate_text

def gpu_warmup(device='cuda'):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch_device = torch.device(device)
    for _ in range(10):
        # dummy calculation
        warmup_tensor = torch.randn((1000, 1000), device=torch_device)
        _ = torch.matmul(warmup_tensor, warmup_tensor)
    torch.cuda.synchronize(torch_device)
    print(f"GPU warmup completed on device: {device}")

model_config = {
    "n_layer": 12,
    "n_embd": 768,
    "n_head": 12,
    "n_positions": 1024,
    "attn_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "resid_pdrop": 0.1,
    "layer_norm_epsilon": 1e-5,
}

input_text = "This is some text"
num_runs = 6
seq_lengths = list(range(50, 1601, 50))

results = []
gpu_warmup()

with open("inference_times_seq.csv", "w", newline="") as csvfile:
    fieldnames = ["seq_len", "n_layer", "n_embd", "n_head", "n_positions", "mean_inf_time", "std_dev_inf_time"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for seq_len in seq_lengths:
        _, mean_inf_time, std_dev_inf_time = generate_text(input_text, seq_len, model_config, num_runs)
        print(f'seq_len: {seq_len}, mean: {mean_inf_time}, std: {std_dev_inf_time}')
        result = {
            "seq_len": seq_len,
            "n_layer": model_config["n_layer"],
            "n_embd": model_config["n_embd"],
            "n_head": model_config["n_head"],
            "n_positions": model_config["n_positions"],
            "mean_inf_time": mean_inf_time,
            "std_dev_inf_time": std_dev_inf_time
        }

        writer.writerow(result)
        print(f'{seq_len} completed')