from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

resp = client.chat.completions.create(
    model="opus-sae-gemma",
    messages=[{"role": "user", "content": "你好，请展示你的代码能力"}]
)
print(resp.choices[0].message.content)