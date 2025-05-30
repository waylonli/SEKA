from pastalib.pasta import PASTA, read_head_config
from transformers import AutoModelForCausalLM,AutoTokenizer

# Initialize pre-trained LLM
name = "/mnt/data/models/Qwen3-8B-Base"
model = AutoModelForCausalLM.from_pretrained(
    name,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")

# Select the attention heads to be steered, 
# following the format of {'layer_id': [head_ids]}: 
# head_config = {
#     "3": [17, 7, 6, 12, 18], "8": [28, 21, 24], "5": [24, 4], 
#     "0": [17], "4": [3], "6": [14], "7": [13], "11": [16], 
# }
head_config = read_head_config("pastalib/config/synthetic_diff0.10/Qwen3-8B-Base_head_config.json")

# Initialize the PASTA steerer
pasta = PASTA(
    model=model,
    tokenizer=tokenizer,
    head_config=head_config, 
    alpha=0.01, # scaling coefficient
    scale_position="exclude", # downweighting unselected tokens
)

# Model Input 
texts = ["Mary is a doctor. She obtains her bachelor degree from UCSD. Answer the occupation of Mary and generate the answer as json format."]

# ===== Without PASTA =====
inputs = tokenizer(texts, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
# ---------------------
# ["The answer should be in json format."]  # returns answer in the wrong format

# ===== With PASTA =====
inputs, offset_mapping = pasta.inputs_from_batch(texts)
# User highlights specific input spans
emphasized_texts = ["Answer the occupation of Mary and generate the answer as json format"]
# PASTA registers the pre_forward_hook to edit attention
with pasta.apply_steering(
    model=model, 
    strings=texts, 
    substrings=emphasized_texts, 
    model_input=inputs, 
    offsets_mapping=offset_mapping
) as steered_model: 
    outputs = steered_model.generate(**inputs, max_new_tokens=128) #, output_attentions=True)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
# -------------------------------
# ['{"name": "Mary", "occupation": "Doctor", ...}']  # returns answer in the correct format