from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from datetime import datetime

# Select device: use MPS if available, else CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Implement Quantization
print("START", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# 8-bit quantization is not supported on MPS, use 4-bit if needed
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": device},
    quantization_config=bnb_config
)

print("DONE", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

##check the datatype and memory of the model
param_dtypes = [param.dtype for param in quantized_model.parameters()]
print(param_dtypes)
print(quantized_model.get_memory_footprint())


##load tokenizer
print("START AutoTokenizer", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
tokenizer = AutoTokenizer.from_pretrained(model_name)
user_input = tokenizer("Portugal  is", return_tensors="pt")
user_input = {k: v.to(device) for k, v in user_input.items()}
print("END AutoTokenizer", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


print("quantized_model START response", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
with torch.no_grad():
    response = quantized_model.generate(**user_input, max_new_tokens=50)
print(tokenizer.batch_decode(response, skip_special_tokens=True))
print("quantized_model END response", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
