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