#!/usr/bin/env python3
"""
Cross-domain null mean transfer — averaged over source domains.

For each (method, model):
  - For every source domain, compute the null mean = mean of AI scores
    (after negation for imbd/l2d so that human scores > AI scores).
  - For every target domain, shift all scores by the source null mean
    (no additional demeaning) and run the knockoff filter.
  - Average FDR, power, frac+, and KS over all source domains
    (excluding the target domain itself).

Output:
  results/cross_domain_null_mean_avg_{method}_{model}.json
  cross_domain_null_mean_avg2_summary.txt
"""

import io
import os
import json
import sys
import numpy as np
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from knockoff_filter import fdr_power, symmetry_check

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")

DOMAINS = [
    'AcademicResearch', 'ArtCulture', 'Business', 'EducationMaterial', 'Entertainment',
    'Environmental', 'Finance', 'FoodCusine', 'GovernmentPublic', 'LegalDocument',
    'MedicalText', 'NewsArticle', 'OnlineContent', 'PersonalCommunication', 'ProductReview',
    'Religious', 'Sports', 'TechnicalWriting', 'TravelTourism',
]

MODELS = ['GPT-3-Turbo', 'GPT-4o', 'Gemini-1.5-Pro', 'Llama-3-70B']
Q_LEVELS = [0.1, 0.2, 0.3, 0.5]

# negate=True: flip sign so human scores are larger (uses the negative statistics)
METHOD_CONFIG = {
    'imbd':       {'raw_suffix': 'imbd_knockoff',       'negate': True},
    'l2d':        {'raw_suffix': 'l2d',                 'negate': True},
    'likelihood': {'raw_suffix': 'likelihood_knockoff', 'negate': False},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scores(domain, model, cfg):
    path = os.path.join(RESULT_DIR, f"{domain}_{model}.{cfg['raw_suffix']}.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    key = 'signed_predictions' if 'signed_predictions' in data else 'predictions'
    human = np.array(data[key]['real'],    dtype=float)
    ai    = np.array(data[key]['samples'], dtype=float)
    if cfg['negate']:
        human = -human
        ai    = -ai
    return human, ai


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment():
    all_results = {}

    for method, cfg in METHOD_CONFIG.items():
        print(f"\n{'='*60}")
        print(f"Method: {method.upper()}")
        print(f"{'='*60}")

        for model in MODELS:
            print(f"\n  Model: {model}")

            # Load all domain scores and null means
            domain_human, domain_ai, domain_null_mean = {}, {}, {}
            for d in DOMAINS:
                h, a = load_scores(d, model, cfg)
                if h is not None:
                    domain_human[d] = h
                    domain_ai[d]    = a
                    domain_null_mean[d] = float(np.mean(a))  # AI null mean

            # For each target domain, collect metrics from all source domains
            model_results = {}
            for tgt in DOMAINS:
                if tgt not in domain_human:
                    continue

                tgt_human = domain_human[tgt]
                tgt_ai    = domain_ai[tgt]
                combined  = np.concatenate([tgt_human, tgt_ai])
                labels    = np.array([1]*len(tgt_human) + [0]*len(tgt_ai), dtype=int)

                per_q = {str(q): {'fdr': [], 'power': [], 'frac': [], 'ks': [], 'ks_pval': []}
                         for q in Q_LEVELS}

                for src, src_mean in domain_null_mean.items():
                    if src == tgt:
                        continue

                    shifted    = combined  - src_mean
                    ai_shifted = tgt_ai   - src_mean
                    sym = symmetry_check(ai_shifted)

                    for q in Q_LEVELS:
                        res = fdr_power(shifted, labels, float(q))
                        per_q[str(q)]['fdr'].append(res['actual_fdr'])
                        per_q[str(q)]['power'].append(res['power'])
                        per_q[str(q)]['frac'].append(sym['fraction_positive'])
                        per_q[str(q)]['ks'].append(sym['ks_statistic'])
                        per_q[str(q)]['ks_pval'].append(sym['ks_pvalue'])

                # Average across source domains
                avg = {}
                for q in Q_LEVELS:
                    vals = per_q[str(q)]
                    avg[str(q)] = {
                        'avg_fdr':     float(np.mean(vals['fdr'])),
                        'avg_power':   float(np.mean(vals['power'])),
                        'avg_frac':    float(np.mean(vals['frac'])),
                        'avg_ks':      float(np.mean(vals['ks'])),
                        'avg_ks_pval': float(np.mean(vals['ks_pval'])),
                        'n_sources':   len(vals['fdr']),
                    }

                model_results[tgt] = avg
                print(f"    {tgt}: done ({avg[str(Q_LEVELS[0])]['n_sources']} sources)")

            all_results.setdefault(method, {})[model] = model_results

            out_path = os.path.join(RESULT_DIR,
                                    f"cross_domain_null_mean_avg_{method}_{model}.json")
            with open(out_path, 'w') as f:
                json.dump(model_results, f, indent=2)
            print(f"    Saved: {out_path}")

    return all_results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

MODEL_GROUPS = {
    'GPT-3.5': ['GPT-3-Turbo'],
    'GPT-4o':  ['GPT-4o'],
    'Llama':   ['Llama-3-70B'],
    'Gemini':  ['Gemini-1.5-Pro'],
}


def print_summary(all_results):
    lines = []
    grp_names = list(MODEL_GROUPS.keys())
    col_w = 11

    header = (
        f"\n{'='*110}\n"
        f"Cross-Domain Null Mean Transfer  —  averaged over all source domains\n"
        f"(imbd/l2d use negated scores; likelihood uses raw scores)\n"
        f"Sorted by average power across model groups\n"
        f"{'='*110}"
    )
    print(header)
    lines.append(header)

    col_header = (
        f"  {'Target Domain':<24}"
        + "".join(
            f"  {g+' FDR':>{col_w}}{g+' Pwr':>{col_w}}{g+' Fr+':>{col_w}}{g+' KS':>{col_w}}"
            for g in grp_names
        )
        + f"  {'AvgPwr':>8}"
    )
    sep = "  " + "-" * 110

    for method in METHOD_CONFIG:
        if method not in all_results:
            continue
        for q in Q_LEVELS:
            block_header = f"\nMethod={method.upper()}  q={q}"
            print(block_header)
            lines.append(block_header)
            print(col_header)
            print(sep)
            lines.extend([col_header, sep])

            tgt_rows = []
            for tgt in DOMAINS:
                grp_stats = {}
                for g, model_list in MODEL_GROUPS.items():
                    vals = [
                        all_results[method].get(m, {}).get(tgt, {}).get(str(q))
                        for m in model_list
                    ]
                    vals = [v for v in vals if v is not None]
                    grp_stats[g] = (
                        {k: np.mean([v[f'avg_{k}'] for v in vals])
                         for k in ('fdr', 'power', 'frac', 'ks')}
                        if vals else None
                    )

                valid_pwrs = [s['power'] for s in grp_stats.values() if s is not None]
                if not valid_pwrs:
                    continue
                tgt_rows.append((tgt, grp_stats, float(np.mean(valid_pwrs))))

            tgt_rows.sort(key=lambda x: x[2])

            for tgt, grp_stats, avg_pwr in tgt_rows:
                row = f"  {tgt:<24}"
                for g in grp_names:
                    s = grp_stats[g]
                    if s is not None:
                        row += (
                            f"  {s['fdr']:>{col_w}.3f}"
                            f"{s['power']:>{col_w}.3f}"
                            f"{s['frac']:>{col_w}.3f}"
                            f"{s['ks']:>{col_w}.3f}"
                        )
                    else:
                        row += f"  {'--':>{col_w}}{'--':>{col_w}}{'--':>{col_w}}{'--':>{col_w}}"
                row += f"  {avg_pwr:>8.3f}"
                print(row)
                lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_experiment()

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_summary(results)
    summary_text = buf.getvalue()
    print(summary_text)

    summary_path = os.path.join(
        os.path.dirname(__file__), "cross_domain_null_mean_avg2_summary.txt"
    )
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"\nSummary saved: {summary_path}")
