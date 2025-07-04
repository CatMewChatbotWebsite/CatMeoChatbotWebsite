from together import Together
from api_key import load_together_apikey

api_key = load_together_apikey()

client = Together(api_key=api_key, timeout=10)

def call_api_llamavision(prompt):
  try:
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",             #https://api.together.ai/models
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
    print("❌ Lỗi call_api_llm:", e, flush=True)


