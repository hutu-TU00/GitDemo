# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM  # AutoModelForCausalLM

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
model = AutoModelForCausalLM.from_pretrained("uer/gpt2-chinese-cluecorpussmall")


prompt = "如果世界上只剩下我一个人。"

input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(
    input_ids,
    max_length=1000,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9,
    num_return_sequences=1
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("生成结果：")
print(generated_text)