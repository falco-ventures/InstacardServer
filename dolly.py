import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

generate_text = pipeline(
                         model="databricks/dolly-v2-12b", 
                         torch_dtype=torch.bfloat16, 
                         trust_remote_code=True, 
                         device_map="auto")

res = generate_text("Explain to me the difference between nuclear fission and fusion.")
print(res[0]["generated_text"])

import torch
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", device_map="auto", torch_dtype=torch.bfloat16)

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
