#!/usr/bin/env python3
"""
Average cross_domain_null_mean_avg results across all target domains.

Reads:  results/cross_domain_null_mean_avg_{method}_{model}.json
Prints: one row per model group per method/q, averaged over all target domains.
"""

import os
import json
import numpy as np

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
MODELS     = ['GPT-3-Turbo', 'GPT-4o', 'Gemini-1.5-Pro', 'Llama-3-70B']
METHODS    = ['imbd', 'l2d', 'likelihood']
Q_LEVELS   = [0.1, 0.2, 0.3, 0.5]

MODEL_GROUPS = {
    'GPT-3.5': ['GPT-3-Turbo'],
    'GPT-4o':  ['GPT-4o'],
    'Llama':   ['Llama-3-70B'],
    'Gemini':  ['Gemini-1.5-Pro'],
}


def load(method, model):
    path = os.path.join(RESULT_DIR, f"cross_domain_null_mean_avg_{method}_{model}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


lines = []

header = (
    f"\n{'='*90}\n"
    f"Cross-Domain Null Mean Transfer — overall average across all target domains\n"
    f"{'='*90}"
)
print(header)
lines.append(header)

col_header = (
    f"  {'Model Group':<14}"
    f"  {'Avg FDR':>10}  {'Avg Power':>10}  {'Avg Frac+':>10}  {'Avg KS':>10}"
)
sep = "  " + "-" * 60

for method in METHODS:
    for q in Q_LEVELS:
        block = f"\nMethod={method.upper()}  q={q}"
        print(block)
        lines.append(block)
        print(col_header)
        print(sep)
        lines.extend([col_header, sep])

        for grp, model_list in MODEL_GROUPS.items():
            fdrs, pwrs, fracs, kss = [], [], [], []
            for model in model_list:
                data = load(method, model)
                if data is None:
                    continue
                for tgt, tgt_data in data.items():
                    entry = tgt_data.get(str(q))
                    if entry is None:
                        continue
                    fdrs.append(entry['avg_fdr'])
                    pwrs.append(entry['avg_power'])
                    fracs.append(entry['avg_frac'])
                    kss.append(entry['avg_ks'])

            if not fdrs:
                row = f"  {grp:<14}  {'--':>10}  {'--':>10}  {'--':>10}  {'--':>10}"
            else:
                row = (
                    f"  {grp:<14}"
                    f"  {np.mean(fdrs):>10.3f}"
                    f"  {np.mean(pwrs):>10.3f}"
                    f"  {np.mean(fracs):>10.3f}"
                    f"  {np.mean(kss):>10.3f}"
                )
            print(row)
            lines.append(row)

summary = "\n".join(lines)
out = os.path.join(os.path.dirname(__file__), "cross_domain_null_mean_overall_summary.txt")
with open(out, 'w') as f:
    f.write(summary)
print(f"\nSaved: {out}")
