import os
import sys
import time
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from langchain_groq import ChatGroq
from evaluation.dataset import load_dataset
from evaluation.evaluator import BenchmarkEvaluator
from agentic_rag_sop.generation.chain import SOPRAGChain

qwen_llm = ChatGroq(model_name='qwen/qwen3.8-27b', groq_api_key=os.getenv('GROQ_API_KEY'))
chain = SOPRAGChain(llm=qwen_llm)
evaluator = BenchmarkEvaluator(chain=chain, delay_seconds=0.0)

dataset = load_dataset()
q = dataset[0]
print(f"Testing Qwen backbone on {q['id']} ({q['category']})...")

t0 = time.time()
res = evaluator.evaluate_single_question(
    q,
    systems_to_run=['llm_only', 'traditional_rag', 'graph_rag', 'hybrid_rag', 'proposed_agentic'],
    run_ablations=False
)
elapsed = time.time() - t0
print(f"Single question evaluation with Qwen finished in {elapsed:.2f}s")
for sys_name, data in res['baselines'].items():
    print(f"  {sys_name}: faith={data.get('faithfulness')}, bert={data.get('bert_score')}, err={data.get('error')}")
