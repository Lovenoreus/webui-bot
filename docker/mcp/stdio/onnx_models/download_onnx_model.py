import os
import requests

MODEL_NAME = "all-MiniLM-L6-v2"
BASE_URL = f"https://huggingface.co/Xenova/{MODEL_NAME}/resolve/main"

FILES = [
    "onnx/model.onnx",
    "onnx/config.json",
    "tokenizer.json",
    "vocab.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
]

script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(script_dir, MODEL_NAME)
os.makedirs(target_dir, exist_ok=True)

print(f"[INFO] Downloading ONNX model to: {target_dir}\n")

for filename in FILES:
    url = f"{BASE_URL}/{filename}"
    out_path = os.path.join(target_dir, os.path.basename(filename))

    try:
        print(f"Downloading {filename} ...", end=" ")
        r = requests.get(url, allow_redirects=True, timeout=120, verify=False)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        size_mb = len(r.content) / (1024 * 1024)
        print(f"done ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"failed: {e}")

print("\n[INFO] All downloads complete.")
print(f"[INFO] Files saved in: {target_dir}")
