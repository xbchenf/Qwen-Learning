# vllm_openai_completions.py
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-xxx", # 随便填写，只是为了通过接口参数校验
)

completion = client.chat.completions.create(
  model="Qwen2-7B-Instruct",
  messages=[
    {"role": "user", "content": "你好"}
  ]
)

print(completion.choices[0].message)

print(10/0)