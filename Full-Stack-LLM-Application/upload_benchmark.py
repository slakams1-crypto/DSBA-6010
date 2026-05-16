import os
from dotenv import load_dotenv
from langsmith import Client
from benchmark_data import MEDICAL_QA_BENCHMARK

load_dotenv()

client = Client()

# Create the dataset in LangSmith
dataset = client.create_dataset(
    "medical-qa-benchmark-v1",
    description="20-example MedIntel benchmark for evaluation"
)

# Upload each example
for ex in MEDICAL_QA_BENCHMARK:
    client.create_example(
        inputs={"input": ex["input"]},
        outputs={"answer": ex["answer"], "context": ex["context"]},
        dataset_id=dataset.id,
    )

print(f"✅ Uploaded {len(MEDICAL_QA_BENCHMARK)} examples.")
print(f"🔗 View in LangSmith: https://smith.langchain.com/public/{dataset.id}/datasets")