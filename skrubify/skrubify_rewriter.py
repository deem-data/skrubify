import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

from skrubify.load_prompts import load_prompts


class Skrubify:
    def __init__(self):
        load_dotenv(override=True)
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.gcp_client = genai.Client()
        self.SYSTEM_PROMPTS = load_prompts()
        self.mode = 5

    def rewrite(self, pipeline, model, mode):
        t0 = time.time()
        user_input = f"Rewrite this pipeline:\n\n```python\n{pipeline}\n```"
        if model.startswith("gpt"):
            response = self.send_gpt(self.SYSTEM_PROMPTS[mode], user_input, model)
        elif model.startswith("gemini"):
            response = self.send_gcp(self.SYSTEM_PROMPTS[mode], user_input, model)
        else:
            print("Cant find model provider for:", model)
            response = "EMPTY"
        t1 = time.time()
        print(f"LLM inference time: {t1 - t0:.2f} seconds")
        return response

    def send_gpt(self, system_prompt, user_input, model):
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content":user_input},]
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=0 if model == "gpt-4.1" else 1,
        )
        return response.choices[0].message.content


    def send_gcp(self, system_prompt, user_input, model="gemini-2.5-pro"):
        # Concatenate system and user prompts to simulate roles
        prompt = f"System: {system_prompt}\nUser: {user_input}"

        response = self.gcp_client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text