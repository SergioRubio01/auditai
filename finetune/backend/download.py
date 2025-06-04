from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
import torch
import os
from dotenv import load_dotenv

load_dotenv()

# Define local directory and repo
local_dir = "pagos"
repo_id = "SergioAeroAI/pagos_v2"

# Load the model state dict from the safetensors files
model_state_dict = {}
model_files = [
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors"
]

# Load all safetensors model parts
for path in model_files:
    model_state_dict.update(load_file(os.path.join(local_dir, path)))

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_dir)

# Create the model and load the state dict
model = AutoModelForCausalLM.from_pretrained(local_dir)
model.load_state_dict(model_state_dict)

# Save the model as a PyTorch model if needed
torch.save(model.state_dict(), "pagos/pytorch_model.bin")

print("Model loaded and saved successfully!")
