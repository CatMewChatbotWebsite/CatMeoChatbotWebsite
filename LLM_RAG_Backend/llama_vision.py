from together import Together
from api_key import apikey

api_key = apikey()

client = Together(api_key=api_key)
def call_api_llm(prompt):
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
<<<<<<< HEAD
    return "❌ Lỗi khi gọi model"
=======
    return "❌ Lỗi khi gọi model"
>>>>>>> 26a0586a836148ea178f4ba733f511c1a84ef9e0
