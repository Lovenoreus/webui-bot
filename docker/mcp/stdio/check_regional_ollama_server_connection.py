# import requests

OLLAMA_HOST = "http://vs2153.vll.se:11434"  # replace with your Ollama server address
MODEL = "mistral:7b"

# r = requests.post(f"{OLLAMA_HOST}/api/generate", json={
#     "model": MODEL,
#     "prompt": "Hello!"
# })

# print(r.status_code)
# print(r.text)

import requests
import time

success = 0
total = 100

for i in range(total):
    print(i)
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/generate", json={
            "model": MODEL,
            "prompt": "ping"
        }, timeout=10)
        if r.status_code == 200:
            success += 1
        else:
            print(f"Ping {i+1} failed: {r.status_code}")
    except Exception as e:
        print(f"Ping {i+1} error: {e}")
    time.sleep(0.5)  # small delay to avoid overload

print(f"\nSuccessful pings: {success}/{total} ({success/total*100:.1f}% success rate)")

