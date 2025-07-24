from together import Together
from openai import OpenAI
from api_key import load_together_apikey

api_key = load_together_apikey()

client = Together(api_key=api_key, timeout=20)

def call_api_llamavision(prompt):
  try:
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",             #https://api.together.ai/models
        messages=[
          {
            "role": "user",
            "content": prompt
          }
        ]
    )
  #print(f"🤖 Trợ lý -> {response.choices[0].message.content}")
    return response.choices[0].message.content
  except Exception as e:
    print(f"Lỗi call llm api: {e}", flush=True)
  return "⚠️ Server đang quá tải, vui lòng thử lại sau"


def call_api_llm_qwen3_coder(prompt):
  try:
    completion = client.chat.completions.create(
    model="deepseek/deepseek-r1-0528-qwen3-8b:free",
    messages=[
        {
        "role": "user",
        "content": prompt
        }
    ]
    )
    return completion.choices[0].message.content
  except:
    print("⚠️ Server đang quá tải, vui lòng thử lại sau.", flush=True)
  


