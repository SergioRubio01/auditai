from unsloth import FastVisionModel
from unsloth.save import save_to_gguf
import torch
import os
from config import MODEL_NAME  # Import just what we need
import json
from datasets import Dataset


# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/llama-3.2-11b-vision-instruct-bnb-4bit",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = False, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = False, # False if not finetuning MLP layers

    r = 4,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 4,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

# Replace the web dataset loading with local dataset
dataset = Dataset.load_from_disk("dataset")  # Update path as needed

instruction = "You are an expert in processing bank transfer documents. Extract and structure the information you see in this image."

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : f"data:image/png;base64,{sample['image']}"} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["raw_data"]} ]
        },
        { "role" : "user",
          "content" : [
            {"type" : "text",  "text"  : "Please extract the desired information from the image."} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : json.dumps(sample["desired_data"], ensure_ascii=False)} ]
        },
    ]
    return { "messages" : conversation }
pass

converted_dataset = [convert_to_conversation(sample) for sample in dataset]

from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 5,
        # num_train_epochs = 12, # Set this instead of max_steps for full training runs
        learning_rate = 1e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 1,
        max_seq_length = 20000,
    ),
)

# Add before training:
torch.cuda.empty_cache()  # Clear CUDA cache

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved")

trainer_stats = trainer.train()

# After training is complete:

from dotenv import load_dotenv
load_dotenv()

# Remove the existing model directory if it exists
import shutil
model_save_path = "SergioAeroAI/pagos_v2"
if os.path.exists(model_save_path):
    shutil.rmtree(model_save_path)

# 1. Save the tokenizer (modified approach)
tokenizer.save_pretrained(
    "SergioAeroAI/pagos_v2",
    push_to_hub=True,
    token=os.getenv("HF_TOKEN")
)

# 2. Save the merged model (includes adapter)
model.save_pretrained_merged(
    save_directory="SergioAeroAI/pagos_v2",
    push_to_hub=True,
    token=os.getenv("HF_TOKEN"),
    save_method="merged_16bit",  # Must be 16bit for GGUF conversion
    
)

# 3. Optional: Save to GGUF format for Ollama compatibility
# model.save_pretrained_gguf(
#     save_directory="SergioAeroAI/pagos_v2",
#     tokenizer=tokenizer,
#     quantization_method="q4_k_m",  # Recommended quantization
#     push_to_hub=True,
#     token=os.getenv("HF_TOKEN")
# )
# save_to_gguf(
#     model_type = "llama",
#     model_dtype = "float16",
#     model_directory = "SergioAeroAI/pagos_v2",
#     quantization_method = "q4_k_m"
# )

used_memory = round(torch.cuda.max_memory_reserved()/1024/1024/1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory/max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100,3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Save model in GGUF format
# print("Saving model in GGUF format")

# This throws an error:

# from unsloth.save import save_to_gguf
# save_to_gguf(model_type = "llama", model_dtype = "float16", model_directory = MODEL_NAME, quantization_method = "q4_k_m")
# print("Model saved in GGUF format")

# Try this:

# model.save_from_pretrained_merged(
#     save_directory = MODEL_NAME,
#     save_method = "merged_16bit",
#     push_to_hub = True,
# )
