

# 案例3：使用开源大语言模型Qwen2.5搭建RAG与Agent
## 案例简介
基于Qwen2.5本地部署，完成**推理优化加速、对话机器人搭建、LangChain-Chain、LangChain-Agent、RAG文本加载与分割、向量数据库、RAG向量化与优化**全流程实践。

## 案例目的
1. 理解RAG和Agent的原理与开发流程
2. 掌握大语言模型构建完整应用的方法
3. 了解大模型开发中的优化策略

## 案例准备
### 1. 环境准备
```bash
# 创建虚拟环境
conda create -n RAG python=3.11.9
# 激活环境
conda activate RAG
```

### 2. 安装依赖
```bash
# CPU版torch
pip install torch==2.6.0
# GPU版torch：根据CUDA版本自行安装

pip install transformers==4.45.0
pip install modelscope==1.18.1
```

### 3. 下载Qwen模型
新建`downloador.py`：
```python
from modelscope.hub.snapshot_download import snapshot_download

llm_model_dir = snapshot_download(
    'Qwen/Qwen2.5-0.5B-Instruct',
    cache_dir='models'
)
```

### 4. 模型推理基础
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("models/Qwen/Qwen2___5-0___5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "models/Qwen/Qwen2___5-0___5B-Instruct"
).to(device)

prompt = "你好"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(device)
generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)

generated_ids = [
    output_ids[len(input_ids):] 
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

## 案例实现
### 1. Ollama部署
#### 1.1 自定义GGUF模型
创建`Modelfile`：
```modelfile
FROM qwen2.5-7b-instruct-q5_0.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER repeat_penalty 1.05
PARAMETER top_k 20

TEMPLATE """{{ if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{ .System }}
{{- if .Tools }}
# Tools
You are provided with function signatures within <tools></tools> XML tags:
<tools>{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}{{- end }}
</tools>
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""

SYSTEM """You are Qwen, created by Alibaba Cloud. You are a helpful assistant."""
```

创建并运行模型：
```bash
ollama create qwen2.5_7b -f Modelfile
ollama run qwen2.5_7b
```

### 2. vLLM推理加速（选学）
```bash
pip install vllm==0.6.3.post1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```python
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "models/Qwen/Qwen2___5-0___5B-Instruct",
    trust_remote_code=True
)

sampling_params = SamplingParams(
    temperature=0.5,
    top_p=0.5,
    repetition_penalty=1.05,
    max_tokens=512
)

prompt = "你好"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

llm = LLM(model="models/Qwen/Qwen2___5-7B-Instruct", trust_remote_code=True)
outputs = llm.generate([text], sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

#### 2.3 OpenAI兼容服务
```bash
python -m vllm.entrypoints.openai.api_server \
--port 10222 \
--model ~/workdir/models/Qwen/Qwen2___5-7B-Instruct \
--served-model-name Qwen2___5-7B-Instruct
```

### 3. OpenAI风格访问Ollama
```bash
pip install openai==1.71.0
```

```python
from openai import OpenAI

api_key = 'ollama'
base_url = 'http://localhost:11434/v1'
client = OpenAI(api_key=api_key, base_url=base_url)

response = client.chat.completions.create(
    model='qwen2.5:0.5b',
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好"}
    ],
    max_tokens=150,
    temperature=0.7,
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end='')
```

### 4. FastAPI后端服务
```bash
pip install fastapi==0.115.12 uvicorn==0.34.0
```

```python
from fastapi import FastAPI, Body
from openai import AsyncOpenAI
from typing import List
from fastapi.responses import StreamingResponse

app = FastAPI()

api_key = 'ollama'
base_url = 'http://localhost:11434/v1'
aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)

messages = []

@app.post("/chat")
async def chat(
    query: str = Body(..., description="用户输入"),
    sys_prompt: str = Body("你是一个有用的助手。", description="系统提示词"),
    history: List = Body([], description="历史对话"),
    history_len: int = Body(1, description="保留历史对话轮数"),
    temperature: float = Body(0.5, description="LLM采样温度"),
    top_p: float = Body(0.5, description="LLM采样概率"),
    max_tokens: int = Body(None, description="最大token数量")
):
    global messages
    if history_len > 0:
        history = history[-2 * history_len:]
    
    messages.clear()
    messages.append({"role": "system", "content": sys_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": query})

    response = await aclient.chat.completions.create(
        model="qwen2.5:0.5b",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True
    )

    async def generate_response():
        async for chunk in response:
            chunk_msg = chunk.choices[0].delta.content
            if chunk_msg:
                yield chunk_msg

    return StreamingResponse(generate_response(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6066, log_level="info")
```

启动：`python fastapi.py`  
文档：`http://127.0.0.1:6066/docs`

### 5. Streamlit前端
```bash
pip install streamlit==1.39.0
```

```python
import streamlit as st
import requests

backend_url = "http://127.0.0.1:6066/chat"
st.set_page_config(page_title="ChatBot", page_icon="🤖", layout="centered")
st.title("🤖 聊天机器人")

def clear_chat_history():
    st.session_state.history = []

with st.sidebar:
    st.title("ChatBot")
    sys_prompt = st.text_input("系统提示词：", value="You are a helpful assistant.")
    history_len = st.slider("保留历史对话数量：", 1, 10, 1)
    temperature = st.slider("temperature：", 0.01, 2.0, 0.5, 0.01)
    top_p = st.slider("top_p：", 0.01, 1.0, 0.5, 0.01)
    max_tokens = st.slider("max_tokens：", 256, 4096, 1024, 8)
    stream = st.checkbox("stream", value=True)
    st.button("清空聊天历史", on_click=clear_chat_history)

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("来和我聊天~~~"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    data = {
        "query": prompt,
        "sys_prompt": sys_prompt,
        "history_len": history_len,
        "history": st.session_state.history,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }

    response = requests.post(backend_url, json=data, stream=True)
    if response.status_code == 200:
        chunks = ""
        assistant_placeholder = st.chat_message("assistant")
        assistant_text = assistant_placeholder.markdown("")
        
        if stream:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                chunks += chunk
                assistant_text.markdown(chunks)
        else:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                chunks += chunk
            assistant_text.markdown(chunks)
        
        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.history.append({"role": "assistant", "content": chunks})
```

启动：`streamlit run streamlit.py`  
访问：`http://localhost:8501`

### 6. Gradio前端
```bash
pip install gradio==5.0.2
```

```python
import gradio as gr
import requests

backend_url = "http://127.0.0.1:6066/chat"

def chat_with_backend(prompt, history, sys_prompt, history_len, temperature, top_p, max_tokens, stream):
    history_clean = [{"role": h.get("role"), "content": h.get("content")} for h in history]
    data = {
        "query": prompt,
        "sys_prompt": sys_prompt,
        "history": history_clean,
        "history_len": history_len,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
    response = requests.post(backend_url, json=data, stream=True)
    chunks = ""
    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            chunks += chunk
            if stream:
                yield chunks
    yield chunks

with gr.Blocks(fill_width=True, fill_height=True) as demo:
    with gr.Tab("🤖 聊天机器人"):
        gr.Markdown("## 🤖 聊天机器人")
        with gr.Row():
            with gr.Column(scale=1, variant="panel"):
                sys_prompt = gr.Textbox(label="系统提示词", value="You are a helpful assistant")
                history_len = gr.Slider(1, 10, 1, label="保留历史对话数量")
                temperature = gr.Slider(0.01, 2.0, 0.5, step=0.01, label="temperature")
                top_p = gr.Slider(0.01, 1.0, 0.5, step=0.01, label="top_p")
                max_tokens = gr.Slider(512, 4096, 1024, step=8, label="max_tokens")
                stream = gr.Checkbox(label="stream", value=True)
            with gr.Column(scale=10):
                chatbot = gr.Chatbot(type="messages", height=500)
                gr.ChatInterface(
                    fn=chat_with_backend,
                    type="messages",
                    chatbot=chatbot,
                    additional_inputs=[sys_prompt, history_len, temperature, top_p, max_tokens, stream]
                )

demo.launch()
```

启动：`python gradio.py`  
访问：`http://127.0.0.1:7860`

### 7. 生成参数说明
| 任务类型 | temperature | top_p | 说明 |
| --- | --- | --- | --- |
| 代码生成 | 0.2 | 0.1 | 输出精确、规范 |
| 创意写作 | 0.7 | 0.8 | 多样性强、创意丰富 |
| 聊天机器人 | 0.5 | 0.5 | 平衡准确与多样 |
| 代码注释 | 0.1 | 0.2 | 简洁准确、贴合代码 |

### 8. LangChain-Chain
#### 8.1 LLMChain
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{user_question}")
])

chat_model = ChatOpenAI(
    openai_api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="qwen2.5:0.5b"
)

chain = chat_prompt | chat_model
response = chain.invoke({"user_question": "你好"})
print(response.content)
```

#### 8.2 RAG检索链
```bash
pip install sentence-transformers==3.3.0 faiss-cpu==1.9.0 langchain-huggingface==0.1.2 langchain-community==0.3.21
```

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

chat_model = ChatOpenAI(
    openai_api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="qwen2.5:0.5b"
)

loader = TextLoader("sanguoyanyi.txt", encoding='utf-8')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name='models/AI-ModelScope/bge-large-zh-v1___5')
vs = FAISS.from_documents(chunks, embedding)
retriever = vs.as_retriever()

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据已知信息回答：{context}"),
    ("user", "问题：{question}")
])

qa = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": chat_prompt}
)

print(qa.invoke("五虎上将有哪些？"))
```

### 8.5 RAG 实战：成语接龙游戏（完整版）
基于 FAISS 向量库 + Ollama 大模型，实现**严格成语接龙**，AI 不会编造成语，百分百从本地文本库匹配。

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import Document  # 导入Document类
import os

# ===================== 路径配置（完全按你的路径） =====================
IDIOM_PATH = r"D:\Large Model\downloador\chengyujielong.txt"
MODEL_PATH = r"D:\Large Model\downloador\AI-ModelScope\bge-large-zh-v1___5"

# ===================== 【终极修复 1：强制读全所有成语，解决27个问题】 =====================
# 第一步：手动读取所有成语，彻底解决TextLoader只读27行的问题
try:
    # 二进制读取，彻底解决编码/换行符问题
    with open(IDIOM_PATH, "rb") as f:
        raw_bytes = f.read()
    # UTF-8解码，忽略错误字符
    content = raw_bytes.decode("utf-8", errors="replace")
    # 按任意换行符分割，过滤空行，去重
    all_lines = content.splitlines()
    all_idioms = [line.strip() for line in all_lines if line.strip()]
    all_idioms = list(set(all_idioms))  # 去重
    idiom_set = set(all_idioms)
    
    print(f"✅ 成功加载成语：{len(all_idioms)} 个（全部内容）")
    print("🔍 前5个成语：", all_idioms[:5])
    print("🔍 后5个成语：", all_idioms[-5:])
    
    if len(all_idioms) == 0:
        print("❌ 错误：文件为空！请检查文件内容！")
        exit()
        
except Exception as e:
    print(f"❌ 读取文件失败：{e}")
    exit()

# ===================== 【终极修复 2：手动构建Document，彻底解决TextLoader问题】 =====================
# 手动把每个成语做成一个Document，绝对不会丢数据
docs = [Document(page_content=idiom) for idiom in all_idioms]
chunks = docs  # 不切割，每个成语一个文档

# ===================== 初始化嵌入模型（正确参数） =====================
embedding = HuggingFaceEmbeddings(model_name=MODEL_PATH)

# ===================== 构建FAISS向量库（用手动构建的docs，绝对读全） =====================
vs = FAISS.from_documents(chunks, embedding)
retriever = vs.as_retriever()

# ===================== 初始化Ollama模型（完全按你的配置） =====================
chat_model = ChatOpenAI(
    openai_api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="qwen2.5:0.5b"
)

# ===================== 成语接龙提示词（严格限制） =====================
system_message = SystemMessagePromptTemplate.from_template("""
你是专业成语接龙AI，必须严格遵守以下规则：
1. 只能使用提供的成语库中的成语，绝对禁止编造。
2. 你接的成语的第一个字，必须等于对方成语的最后一个字。
3. 只输出4字成语，不要任何多余内容、解释、标点。

已知成语库：
{context}
""")

chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    HumanMessagePromptTemplate.from_template("用户成语：{question}，请接龙")
])

# ===================== 构建RetrievalQA链 =====================
qa = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": chat_prompt}
)

# ===================== 游戏主逻辑 =====================
print("="*60)
print("🎮 成语接龙游戏（RAG + FAISS + Ollama 最终完美版）")
print("规则：首尾相接 | 成语必须在文本库中 | 错误即结束")
print("="*60)

last_char = None

while True:
    # 玩家输入
    user_idiom = input("\n👉 请输入成语：").strip()

    # 1. 检查玩家成语是否在库中
    if user_idiom not in idiom_set:
        print("❌ 该成语不在文本库中，游戏结束！")
        break

    # 2. 检查接龙规则（非第一轮）
    if last_char is not None:
        if user_idiom[0] != last_char:
            print(f"❌ 首字必须是：{last_char}，游戏结束！")
            break

    # 3. AI 接龙（双重保险，防止AI乱编）
    try:
        ai_result = qa.invoke(user_idiom)
        ai_idiom = ai_result['result'].strip()
        
        # 强制清洗AI输出，只保留前4个字
        if len(ai_idiom) > 4:
            ai_idiom = ai_idiom[:4]
            
        # 终极防呆：如果AI输出不在库中，强制从库中找一个正确的
        if ai_idiom not in idiom_set:
            target_char = user_idiom[-1]
            # 从库中找第一个以此开头的成语
            for idiom in idiom_set:
                if idiom.startswith(target_char):
                    ai_idiom = idiom
                    break
                    
    except Exception as e:
        print(f"🤖 AI 思考出错，自动从库中匹配...")
        target_char = user_idiom[-1]
        for idiom in idiom_set:
            if idiom.startswith(target_char):
                ai_idiom = idiom
                break

    print(f"🤖 AI 接龙：{ai_idiom}")
    # 更新下一轮的尾字
    last_char = ai_idiom[-1]
```

### 9. LangChain-Agent
```python
import requests
from pydantic import Field
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

class WeatherTool:
    def __init__(self, api_key):
        self.api_key = api_key
    def run(self, city):
        city = city.split("\n")[0]
        url = f"https://api.seniverse.com/v3/weather/now.json?key={self.api_key}&location={city}&language=zh-Hans&unit=c"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather = data["results"][0]["now"]["text"]
            tem = data["results"][0]["now"]["temperature"]
            return f"{city}天气：{weather}，温度：{tem}°C"
        else:
            return f"无法获取{city}天气"

api_key = "SBJVysU9a4KvOtgHs"
weather_tool = WeatherTool(api_key)

tools = [
    Tool(
        name="weather check",
        func=weather_tool.run,
        description="查询城市天气"
    )
]

chat_model = ChatOpenAI(
    api_key="sk-xxx",
    base_url="https://api.siliconflow.cn/v1",
    model="Qwen/Qwen2.5-7B-Instruct"
)

template = """回答问题，可使用工具：
{tools}
格式：
Question: 问题
Thought: 思考
Action: 工具名
Action Input: 输入
Observation: 结果
Thought: 最终答案
Final Answer: 答案

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)
agent = create_react_agent(chat_model, tools, prompt, stop_sequence=["\nObserv"])
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({"input": "成都天气怎么样"})
print(response)
```

### 10. 模型微调（BitFit-Tuning）
#### 10.1 参数冻结
```python
from transformers import AutoModelForCausalLM

model_name = "models/Qwen/Qwen2___5-0___5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)

for name, param in model.named_parameters():
    if "bias" not in name:
        param.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数：{trainable_params}")
```

#### 10.2 法律领域微调
```python
import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from accelerate import Accelerator
import math

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

model_name = "models/Qwen/Qwen2___5-0___5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)

# 冻结非bias参数
for name, param in model.named_parameters():
    if "bias" not in name:
        param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained(model_name)

class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    self.data.append(item)
                except:
                    continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example["input"].split("<问题>：\n")[-1].strip()
        
        system_prompt = "<|im_start|>system\n你是一名法律助手，根据中国法律回答用户问题。<|im_end|>\n"
        user_prompt = f"<|im_start|>user\n{question}<|im_end|>\n"
        assistant_prompt = f"<|im_start|>assistant\n{example['output']}<|im_end|>"
        full_text = system_prompt + user_prompt + assistant_prompt

        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        labels = encoded["input_ids"].clone()
        assistant_start = full_text.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
        labels[:, :assistant_start] = -100

        return {
            'input_ids': encoded["input_ids"].squeeze(),
            'attention_mask': encoded["attention_mask"].squeeze(),
            'labels': labels.squeeze()
        }

dataset = CustomDataset('dataset/Law-QA.jsonl', tokenizer)
batch_size = 4
num_epochs = 500
grad_accum_steps = 8

accelerator = Accelerator()
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 学习率调度
warmup_ratio = 0.1
num_update_steps_per_epoch = len(data_loader) // grad_accum_steps
num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
total_steps = num_update_steps_per_epoch * num_epochs
warmup_steps = int(total_steps * warmup_ratio)

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

optimizer = AdamW(model.parameters(), lr=6e-4, eps=1e-8)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

data_loader, model, optimizer = accelerator.prepare(data_loader, model, optimizer)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for step, batch in enumerate(data_loader):
        outputs = model(**batch)
        loss = outputs.loss / grad_accum_steps
        train_loss += loss.detach() * grad_accum_steps
        accelerator.backward(loss)

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(data_loader):
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    avg_loss = accelerator.gather(train_loss).mean() / len(data_loader)
    if (epoch + 1) % 10 == 0 and accelerator.is_local_main_process:
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    if (epoch + 1) % 100 == 0 and accelerator.is_local_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(f"logs/model_{epoch+1}")
        tokenizer.save_pretrained(f"logs/model_{epoch+1}")
```

#### 10.3 微调模型推理
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "logs/models/Qwen/Qwen2___5-0___5B-Instruct_500"

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "《劳动法》第三十一条是什么？"
messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(device)
generated_ids = model.generate(**model_inputs, max_new_tokens=512)

generated_ids = [
    output_ids[len(input_ids):] 
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("模型回答：\n", response)
```

---

