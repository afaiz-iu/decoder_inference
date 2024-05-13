import torch
import csv
from inference import generate_text
from math import ceil

def gpu_warmup(device='cuda'):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch_device = torch.device(device)
    for _ in range(10):
        warmup_tensor = torch.randn((1000, 1000), device=torch_device)
        _ = torch.matmul(warmup_tensor, warmup_tensor)
    torch.cuda.synchronize(torch_device)
    print(f"GPU warmup completed on device: {device}")

# Define the base model configuration
base_model_config = {
    "n_layer": 6,
    "n_embd": 840,
    "n_head": 4,
    "n_positions": 1024,
    "attn_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "resid_pdrop": 0.1,
    "layer_norm_epsilon": 1e-5,
}

input_text = "This is some text"
num_runs = 30
seq_length = 800

# model configurations
config_variations = {
    "num_layers": [4, 6, 8, 10, 12],
    "num_heads": [4, 6, 8, 10, 12, 14],
    "num_embed": [128, 256, 512, 768, 820, 1024]
}

gpu_warmup()

for config_name, values in config_variations.items():
    results = []
    with open(f"inference_times_{config_name}_800.csv", "w", newline="") as csvfile:
        fieldnames = ["config_value", "mean_inf_time", "std_dev_inf_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for value in values:
            model_config = base_model_config.copy()
            if config_name == "num_layers":
                model_config["n_layer"] = value
            elif config_name == "num_heads":
                model_config["n_head"] = value
            elif config_name == "num_embed":
                model_config["n_embd"] = value

            # n_embd to be divisible by n_heads
            if model_config["n_embd"] % model_config["n_head"] != 0:
                model_config["n_embd"] = ceil(model_config["n_embd"] / model_config["n_head"]) * model_config["n_head"]
                print(f'Adjusted n_embd to {model_config["n_embd"]} for n_head={model_config["n_head"]}')

            _, mean_inf_time, std_dev_inf_time = generate_text(input_text, seq_length, model_config, num_runs)
            print(f'{config_name}: {value}, mean: {mean_inf_time}, std: {std_dev_inf_time}')
            result = {
                "config_value": value,
                "mean_inf_time": mean_inf_time,
                "std_dev_inf_time": std_dev_inf_time
            }

            writer.writerow(result)
            print(f'{config_name}={value} completed')
