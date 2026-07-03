from models.phi3_model import Phi3Model
import time

print("=" * 60)
print("STEP 1: Creating model...")
print("=" * 60)

model = Phi3Model()

print("=" * 60)
print("STEP 2: Model loaded successfully")
print("=" * 60)

question = "What causes diabetes?"

print("=" * 60)
print("STEP 3: Starting generation...")
print("=" * 60)

start = time.time()

result = model.generate(
    question,
    max_new_tokens=200,
    temperature=0.0,
    top_p=1.0
)

end = time.time()

print("=" * 60)
print("STEP 4: Generation finished")
print("=" * 60)

print(f"Time taken: {end-start:.2f} seconds")

print("\nANSWER:")
print(result["answer"])