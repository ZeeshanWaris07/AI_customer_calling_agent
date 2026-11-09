import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import faulthandler
faulthandler.enable() 

import torch

import sys
import traceback

from app.agent import handle_faq_question


test_questions = [
    "Can I send my purchased item back?",
    "How long do I have to return something?",
    "Do you offer free laptop installation?",
]

print("Python executable:", sys.executable, flush=True)
print("Starting RAG answer-generation tests...", flush=True)

for question in test_questions:
    print("\nCustomer:", question, flush=True)

    try:
        print("Calling handle_faq_question()...", flush=True)

        result = handle_faq_question(question)

        print("Function returned successfully!", flush=True)
        print("Result:", repr(result), flush=True)

        answer, outcome = result

        print("Outcome:", outcome, flush=True)
        print("Agent:", answer, flush=True)

    except Exception:
        print("\nAn error occurred:", flush=True)
        traceback.print_exc()

print("\nAll tests completed.", flush=True)
