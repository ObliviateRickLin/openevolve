import os
from openai import OpenAI  # ✅ 新版本推荐方式

# 从环境变量读取 API Key 更安全
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量读取
)

response = client.chat.completions.create(
    model="gpt-4o-mini",  # 或 "gpt-4"
    messages=[
        {"role": "system", "content": "你是个助理。"},
        {"role": "user", "content": "帮我写一段代码示例。"}
    ]
)

# 访问生成的回复内容
print(response.choices[0].message.content)
