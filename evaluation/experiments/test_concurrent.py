import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding='utf-8')

from evaluation.dataset import load_dataset
from evaluation.evaluator import BenchmarkEvaluator

dataset = load_dataset()
evaluator = BenchmarkEvaluator(delay_seconds=0.0)

def eval_q(q):
    t0 = time.time()
    res = evaluator.evaluate_single_question(
        q,
        systems_to_run=['llm_only', 'traditional_rag', 'graph_rag', 'hybrid_rag', 'proposed_agentic'],
        run_ablations=False
    )
    elapsed = time.time() - t0
    print(f"Done {q['id']} ({q['category']}) in {elapsed:.2f}s")
    return res

if __name__ == "__main__":
    t_start = time.time()
    print("Testing 2 concurrent questions...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(eval_q, dataset[i]) for i in [0, 1]]
        results = [f.result() for f in futures]
    total_time = time.time() - t_start
    print(f"Total time for 2 questions concurrently: {total_time:.2f}s")
    for r in results:
        print(f"Q: {r['id']}")
        for sys_name, data in r['baselines'].items():
            print(f"  {sys_name}: faith={data.get('faithfulness')}, bert={data.get('bert_score')}, err={data.get('error')}")
