"""
Example: Basic benchmark before and after fine-tuning.

Run:
    python examples/basic_benchmark.py
"""

from sentence_transformers import SentenceTransformer
import retrieval_guard
from retrieval_guard.reporter import export

# 1. Load your model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Run the generalization benchmark BEFORE fine-tuning
print("Running pre-tuning benchmark...")
baseline = retrieval_guard.benchmark.run(model)
print(baseline.to_json())

# 3. Save the baseline for later comparison
export(baseline, format="json", path="baseline_report.json")
print("\nBaseline saved to baseline_report.json")

# 4. ... (fine-tune your model here) ...

# 5. Load your fine-tuned model and check for regression
# fine_tuned = SentenceTransformer("path/to/fine-tuned-model")
# alert = retrieval_guard.benchmark.compare(fine_tuned, baseline)
# if alert.fired:
#     print("REGRESSION DETECTED!")
#     print(alert.recommendation)
#     export(baseline, alert, format="html", path="regression_report.html")
# else:
#     print("No regression. Safe to deploy.")
