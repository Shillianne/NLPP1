import numpy as np
import tiktoken
import random


encoding = tiktoken.get_encoding("gpt2")
seen = set()
data = np.array([], dtype=np.uint16)
for _ in range(100000):
    a, b = random.randint(0, 8192), random.randint(0, 8192)
    pair = (a, b)
    while pair in seen:        
        a, b = random.randint(0, 8192), random.randint(0, 8192)
        pair = (a, b)
    seen.add(pair)
    text = f"{a} + {b} = {a + b}."
    data = np.hstack([data, np.array(encoding.encode(text) + [encoding.eot_token], dtype=np.uint16)])

np.save("./data/test.npy", data)