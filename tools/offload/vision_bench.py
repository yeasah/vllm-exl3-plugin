"""Client: what an image costs, against a server on :8123. Times the same short prompt with and
without an image (max_tokens 1), best of REPS, so the difference is the vision tower and the
image tokens' prefill. Also asks for a short description, to see the image was understood.
usage: vision_bench.py [SIZE] [REPS]   SIZE is the generated image's side in pixels (default 896)"""
import base64, io, json, sys, time, urllib.request
from PIL import Image, ImageDraw

size = int(sys.argv[1]) if len(sys.argv) > 1 else 896
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
img = Image.new("RGB", (size, size), "white"); d = ImageDraw.Draw(img)
d.ellipse((size // 8, size // 8, size // 2, size // 2), fill="red")
d.rectangle((size // 2, size // 2, 7 * size // 8, 7 * size // 8), fill="blue")
buf = io.BytesIO(); img.save(buf, "PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def ask(with_image, n=1, text="Describe this image in one sentence."):
    content = [{"type": "text", "text": text}]
    if with_image: content.insert(0, {"type": "image_url", "image_url": {"url": url}})
    b = json.dumps({"model": "m", "messages": [{"role": "user", "content": content}], "max_tokens": n,
                    "temperature": 0}).encode()
    r = urllib.request.Request("http://127.0.0.1:8123/v1/chat/completions", b, {"Content-Type": "application/json"})
    t = time.time(); o = json.load(urllib.request.urlopen(r, timeout=600)); return time.time() - t, o

ask(True); ask(False)  # warm up both paths
ti = min(ask(True)[0] for _ in range(reps)); tt = min(ask(False)[0] for _ in range(reps))
_, o = ask(True)
print(f"image {size}px: {o['usage']['prompt_tokens']} prompt tokens; TTFT {1e3 * ti:.0f} ms with image, "
      f"{1e3 * tt:.0f} ms without, image costs {1e3 * (ti - tt):.0f} ms", flush=True)
print("says:", repr(ask(True, 40)[1]["choices"][0]["message"]["content"]), flush=True)
