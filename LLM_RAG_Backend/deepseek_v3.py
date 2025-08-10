from openai import OpenAI
from llama_vision import call_api_llamavision
from api_key import load_openrouter_apikey

api_key = load_openrouter_apikey()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
  timeout=15
)

def call_api_llm(prompt, entries=3):
  for i in range(entries):
    try:
      completion = client.chat.completions.create(
      model="deepseek/deepseek-chat-v3-0324:free", # 685B-parameter
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
  return call_api_llm_qwen3(prompt)

def call_api_llm_qwen3(prompt):
  try:
    completion = client.chat.completions.create(
    model="qwen/qwen3-coder:free",
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
  return call_api_llamavision(prompt)

   