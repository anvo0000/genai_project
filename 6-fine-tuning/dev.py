from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


#load model and tokenizer
# model_name = "meta-llama/Llama-3.1-8B-Instruct"

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name,device_map="cpu")

##check the datatype and memory of the model
# param_dtypes = [param.dtype for param in model.parameters()]
# print(param_dtypes)
# print(model.get_memory_footprint())

##load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
user_input = tokenizer("Portugal  is", return_tensors="pt").to("cpu")

response = model.generate(**user_input, max_new_tokens=50)
print(tokenizer.batch_decode(response, skip_special_tokens=True))

## Implement Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True
)