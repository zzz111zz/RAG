import torch
# 导入transformers库中的AutoModelForCausalLM（用于因果语言模型）和 AutoTokenizer（自动化的分词工具）。
from transformers import AutoModelForCausalLM, AutoTokenizer
# 检查是否有可用的CUDA设备（即NVIDIA GPU），如果有，则设置为使用GPU，否则使用CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 使用预训练模型Qwen2.5-0.5的分词器实例化tokenizer对象。
tokenizer = AutoTokenizer.from_pretrained("D:\Large Model\downloador\Qwen\Qwen2___5-0___5B-Instruct")
# 实例化一个预训练的因果语言模型，并将其移动到指定的device上执行。
model = AutoModelForCausalLM.from_pretrained("D:\Large Model\downloador\Qwen\Qwen2___5-0___5B-Instruct").to(device)
# 设置对话的提示词。
prompt = "你好"
# 定义对话历史，包括系统信息（指示助手的角色）和用户输入的提示。
messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
]
# 使用分词器将对话历史转换为适用于模型输入的格式，并添加生成提示。
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False, 
    add_generation_prompt=True
)
# 对输入文本进行分词处理，并转换为PyTorch张量格式，然后移动到指定的device上执行。
model_inputs = tokenizer([text], return_tensors="pt").to(device)
# 使用模型生成新的token序列，最大生成新的token数量为512。
generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=512,)
# 截取生成的token序列，去除原始输入的部分。
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
# 将生成的token序列解码回文本，并忽略特殊token。
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# 输出生成的响应文本。
print(response)