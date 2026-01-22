# test_basic.py
from models.phi3_model import Phi3Model

model = Phi3Model()
question = "What causes diabetes?"
result = model.generate(question)
print(result['answer'])