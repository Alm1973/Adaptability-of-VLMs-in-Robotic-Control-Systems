import time
from mlx_lm import load, generate

MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

print("Loading model...")

model, tokenizer = load(MODEL)

prompt = "Describe the image: a robot arm picking up an object."


print("\nWarming up...")
for _ in range(2):
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=20)

print("Warmup done.\n")


latencies = []

for i in range(5):
    start = time.perf_counter()

    output = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=20
    )

    end = time.perf_counter()

    latency = end - start
    latencies.append(latency)

    print(f"Trial {i+1}")
    print("Latency:", round(latency, 4))
    print("Output:", output)
    print()

print("====================")
print("FINAL RESULTS (STEADY STATE)")
print("Average:", round(sum(latencies)/len(latencies), 4))
print("Min:", round(min(latencies), 4))
print("Max:", round(max(latencies), 4))
