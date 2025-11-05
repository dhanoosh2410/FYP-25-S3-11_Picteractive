# test_openai_ping.py (replace your current file)
from dotenv import load_dotenv
load_dotenv(".env", override=True)

from openai import OpenAI
import os, sys

print("Key present:", bool(os.getenv("OPENAI_API_KEY")))
client = OpenAI()  # reads OPENAI_API_KEY from env

try:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=3,
    )
    print("OpenAI OK:", resp.choices[0].message.content)
except Exception as e:
    print("OpenAI ERROR:", type(e).__name__, e)
    sys.exit(1)
