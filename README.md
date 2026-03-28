# 基于 Qwen2.5-0.5B 本地大模型的聊天机器人项目完整部署手册

---

## 一、项目整体架构
- **模型**：Qwen2.5-0.5B-Instruct（本地部署）
- **推理服务**：Ollama（提供 OpenAI 兼容接口）
- **后端**：FastAPI（6066 端口，提供流式聊天接口）
- **前端**：Streamlit / Gradio（可视化交互界面）

---

## 二、环境准备
### 1. 安装基础软件
1. 安装 **Python 3.10+**
2. 安装 **Ollama**（官网下载：https://ollama.com/）
3. 安装 **Git**（可选）

### 2. 创建并激活虚拟环境
```bash
# 创建虚拟环境
python -m venv llm_chat

# Windows 激活
llm_chat\Scripts\activate

# Mac/Linux 激活
source llm_chat/bin/activate
```

### 3. 安装依赖包
创建 `requirements.txt`，写入以下内容：
```txt
streamlit
requests
fastapi
uvicorn
openai
gradio
torch
transformers
modelscope
```
执行安装：
```bash
pip install -r requirements.txt
```

---

## 三、下载 Qwen2.5-0.5B 模型
### 1. 新建模型下载脚本
新建 `downloador.py`，粘贴以下代码：
```python
from modelscope.hub.snapshot_download import snapshot_download

# 下载模型到指定目录
llm_model_dir = snapshot_download(
    'Qwen/Qwen2.5-0.5B-Instruct',
    cache_dir="D:\Large Model\downloador"
)
```

### 2. 执行下载
```bash
python downloador.py
```
等待下载完成，模型会保存到 `D:\Large Model\downloador`

---

## 四、用 Ollama 部署本地模型
### 1. 拉取并运行 Qwen 模型
命令行执行：
```bash
ollama pull qwen2.5:0.5b
```

### 2. 验证 Ollama 服务
```bash
ollama list
```
看到 `qwen2.5:0.5b` 即成功。

---

## 五、启动 FastAPI 后端服务
### 1. 新建后端文件
新建 `fastapi1.py`，粘贴以下代码：
```python
from fastapi import FastAPI, Body
from openai import AsyncOpenAI
from typing import List
from fastapi.responses import StreamingResponse

app = FastAPI()

# 对接 Ollama
api_key = 'ollama'
base_url = 'http://localhost:11434/v1'
aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)

@app.post("/chat")
async def chat(
    query: str = Body(..., description="用户输入"),
    sys_prompt: str = Body("你是一个有用的助手。"),
    history: List = Body([], description="历史对话"),
    history_len: int = Body(1),
    temperature: float = Body(0.5),
    top_p: float = Body(0.5),
    max_tokens: int = Body(None)
):
    messages = [{"role": "system", "content": sys_prompt}]
    
    # 控制历史长度
    if history_len > 0:
        history = history[-2 * history_len:]
    messages.extend(history)
    messages.append({"role": "user", "content": query})

    # 流式请求 Ollama
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

### 2. 启动后端
新开终端，激活虚拟环境后执行：
```bash
python fastapi1.py
```
启动成功：**http://127.0.0.1:6066**

---

## 六、启动前端界面（二选一）
### 方式 1：Streamlit 前端（推荐简洁版）
1. 新建 `streamlit.py`
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
        assistant_msg = st.chat_message("assistant")
        assistant_text = assistant_msg.markdown("")

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

2. 启动 Streamlit
```bash
streamlit run streamlit.py
```
访问：**http://localhost:8501**

---

### 方式 2：Gradio 前端（功能丰富版）
1. 新建 `gradio1.py`
```python
import gradio as gr
import requests

backend_url = "http://127.0.0.1:6066/chat"

def chat_with_backend(prompt, history, sys_prompt, history_len, temperature, top_p, max_tokens, stream):
    history_clean = [{"role": h["role"], "content": h["content"]} for h in history]
    data = {
        "query": prompt,
        "sys_prompt": sys_prompt,
        "history": history_clean,
        "history_len": history_len,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
    response = requests.post(backend_url, json=data, stream=stream)
    if response.status_code == 200:
        chunks = ""
        if stream:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                chunks += chunk
                yield chunks
        else:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                chunks += chunk
            yield chunks

with gr.Blocks(fill_width=True, fill_height=True) as demo:
    with gr.Tab("🤖 聊天机器人"):
        gr.Markdown("## 🤖 聊天机器人")
        with gr.Row():
            with gr.Column(scale=1):
                sys_prompt = gr.Textbox(label="系统提示词", value="You are a helpful assistant")
                history_len = gr.Slider(1, 10, 1, label="历史轮数")
                temperature = gr.Slider(0.01, 2.0, 0.5, label="temperature")
                top_p = gr.Slider(0.01, 1.0, 0.5, label="top_p")
                max_tokens = gr.Slider(512, 4096, 1024, label="max_tokens")
                stream = gr.Checkbox(label="stream", value=True)
            with gr.Column(scale=10):
                chatbot = gr.Chatbot(type="messages", height=500)

        gr.ChatInterface(
            fn=chat_with_backend,
            chatbot=chatbot,
            type="messages",
            additional_inputs=[sys_prompt, history_len, temperature, top_p, max_tokens, stream]
        )

demo.launch()
```

2. 启动 Gradio
```bash
python gradio1.py
```
访问：**http://localhost:7860**

---

## 七、本地纯代码推理测试（可选）
新建 `1.py`，直接加载本地模型推理：
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("D:\Large Model\downloador\Qwen\Qwen2___5-0___5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("D:\Large Model\downloador\Qwen\Qwen2___5-0___5B-Instruct").to(device)

prompt = "你好"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
```
运行：
```bash
python 1.py
```

---

## 八、完整运行流程总结
1. 启动 Ollama：`ollama pull qwen2.5:0.5b`
2. 启动 FastAPI：`python fastapi1.py`
3. 启动前端：`streamlit run streamlit.py` 或 `python gradio1.py`
4. 打开浏览器，开始聊天

---

## 九、常见问题
1. **端口被占用**：修改 `fastapi1.py` 中 `port=6066`
2. **模型加载失败**：检查模型路径是否正确
3. **无流式输出**：确认前端 `stream=True`、后端 `stream=True`
4. **Ollama 连接失败**：确认 Ollama 服务正在运行

---
