from langchain.chains.llm import LLMChain
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate,ChatPromptTemplate
from langchain_openai import ChatOpenAI
# 创建一个系统消息，用于定义机器人的角色
system_message = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant."
)
# 创建一个人类消息，用于接收用户的输入
human_message = HumanMessagePromptTemplate.from_template(
    "{user_question}"
)
# 将这些模板结合成一个完整的聊天提示
chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    human_message,
])
# 使用 OpenAI API 的 ChatOpenAI 模型
chat_model = ChatOpenAI(
    openai_api_key="ollama", # ollama兼容OpenAI API的格式
    base_url="http://localhost:11434/v1",
    model="qwen2.5:0.5b"
)
# """旧版本"""
# # 创建一条LLMChain链，verbose=True可以显示提示信息。
# llm_chain = LLMChain(
# llm=chat_model,
# prompt=chat_prompt,
# verbose=True
# )
# # 测试LLMChain
# response = llm_chain("你好")
# print(response["text"])
"""新版本"""
# 创建一个 RunnableSequence 链
chain = chat_prompt | chat_model
# 测试链
response = chain.invoke({"user_question": "你好"})
print(response.content)
