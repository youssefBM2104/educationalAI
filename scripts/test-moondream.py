# test_moondream.py
import httpx
import base64

with open("/tmp/figure_1.png", "rb") as f:
    image_b64 = base64.standard_b64encode(f.read()).decode()

response = httpx.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "moondream",
        "prompt": "Describe this image.",
        "images": [image_b64],
        "stream": False,
    },
    timeout=120.0,
)

print("Raw response:")
print(response.json())