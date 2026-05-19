import os
import json

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")

DOMAINS = [
    'AcademicResearch', 'ArtCulture', 'Business', 'EducationMaterial', 'Entertainment',
    'Environmental', 'Finance', 'FoodCusine', 'GovernmentPublic', 'LegalDocument',
    'MedicalText', 'NewsArticle', 'OnlineContent', 'PersonalCommunication', 'ProductReview',
    'Religious', 'Sports', 'TechnicalWriting', 'TravelTourism',
]
MODELS = {
    'GPT-3-Turbo': 'GPT-3.5T',
    'GPT-4o': 'GPT-4o',
    'Gemini-1.5-Pro': 'Gemini',
    'Llama-3-70B': 'Llama-70B',
}
METHODS = {
    'knockoff_l2d': 'L2D',
    'knockoff_likelihood': 'Likelihood',
    'knockoff_imbd': 'IMBD',
}
# (section, fdr_key) per method
METHOD_SECTIONS = {
    'knockoff_l2d':        ('negative', 'demean_fdr_control'),
    'knockoff_likelihood': ('positive', 'demean_fdr_control'),
    'knockoff_imbd':       ('negative', 'demean_fdr_control'),
}
FDR_LEVELS = ['0.1', '0.2', '0.3', '0.5']


def load_result(domain, model, method_file):
    path = os.path.join(RESULT_DIR, f"{domain}_{model}.{method_file}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def collect_avg(extract_fn):
    """Collect averages across domains for each (model, method).

    extract_fn(result, method_file) -> scalar or None
    """
    avgs = {}
    for model_key in MODELS:
        for method_file in METHODS:
            vals = []
            for domain in DOMAINS:
                result = load_result(domain, model_key, method_file)
                v = extract_fn(result, method_file)
                if v is not None:
                    vals.append(v)
            avgs[(model_key, method_file)] = (
                sum(vals) / len(vals) if vals else None, len(vals)
            )
    return avgs


def collect_avg_multi(extract_fn):
    """Like collect_avg but extract_fn returns a list of values (one per FDR level).

    extract_fn(result, method_file) -> list of scalars or None
    """
    avgs = {}
    for model_key in MODELS:
        for method_file in METHODS:
            accum = None
            count = 0
            for domain in DOMAINS:
                result = load_result(domain, model_key, method_file)
                vs = extract_fn(result, method_file)
                if any(v is not None for v in vs):
                    if accum is None:
                        accum = [0.0] * len(vs)
                    for i, v in enumerate(vs):
                        if v is not None:
                            accum[i] += v
                    count += 1
            if accum is not None and count > 0:
                avgs[(model_key, method_file)] = ([a / count for a in accum], count)
            else:
                avgs[(model_key, method_file)] = (None, 0)
    return avgs


def f2(v):
    return f"{v:.2f}" if v is not None else "---"


def f3(v):
    return f"{v:.3f}" if v is not None else "---"


# ---------------------------------------------------------------------------
# Section 4.1 — Symmetry Condition
# ---------------------------------------------------------------------------

def table_symmetry():
    """
    Shows raw fraction_positive and KS p-value averaged across domains.
    Under Assumption 1, for AI texts fraction_positive should be ~0.5.
    """
    print("=" * 100)
    print("TABLE: Symmetry Condition (Section 4.1)")
    print("  raw fraction_positive: fraction of W_i > 0 over all texts (ideal ~0.75 if symmetry holds")
    print("  for AI texts and power is reasonable; ~0.5 if no detection signal)")
    print("  KS p-value: tests symmetry of W_i distribution (high = more symmetric)")
    print("=" * 100)

    def get_fp(result, method_file):
        if result is None:
            return None
        section = METHOD_SECTIONS[method_file][0]
        return result.get(section, {}).get("symmetry", {}).get("fraction_positive")

    def get_ksp(result, method_file):
        if result is None:
            return None
        section = METHOD_SECTIONS[method_file][0]
        return result.get(section, {}).get("symmetry", {}).get("ks_pvalue")

    fp_avgs  = collect_avg(get_fp)
    ksp_avgs = collect_avg(get_ksp)

    col_w = 10
    model_names = list(MODELS.values())

    # plain-text table
    header = f"  {'Method':<12}" + "".join(
        f"{'frac+':>{col_w}}{'KS-p':>{col_w}}" for _ in MODELS
    )
    subheader = f"  {'':<12}" + "".join(
        f"{m:>{2*col_w}}" for m in model_names
    )
    print(subheader)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for method_file, method_name in METHODS.items():
        row = f"  {method_name:<12}"
        for model_key in MODELS:
            fp, _ = fp_avgs[(model_key, method_file)]
            ksp, _ = ksp_avgs[(model_key, method_file)]
            row += f"{f3(fp):>{col_w}}{f3(ksp):>{col_w}}"
        print(row)

    # LaTeX
    print("\n\n--- LaTeX ---")
    n_models = len(MODELS)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Empirical validation of the symmetry condition (Assumption~1). "
          r"We report the fraction of knockoff statistics $W_i > 0$ (frac$+$) and the "
          r"KS test p-value for symmetry of $W_i$, averaged across domains. "
          r"Values close to 0.5 for frac$+$ indicate the symmetry assumption holds for AI texts.}")
    print(r"\label{tab:symmetry}")
    print(r"\begin{tabular}{l" + "cc" * n_models + "}")
    print(r"\toprule")

    top = ["Method"] + [f"\\multicolumn{{2}}{{c}}{{{m}}}" for m in model_names]
    print(" & ".join(top) + r" \\")

    cmr_parts = []
    for i in range(n_models):
        start = 2 + i * 2
        cmr_parts.append(f"\\cmidrule(lr){{{start}-{start+1}}}")
    print("".join(cmr_parts))

    sub = [" "] + ["frac$+$", "KS-$p$"] * n_models
    print(" & ".join(sub) + r" \\")
    print(r"\midrule")

    for method_file, method_name in METHODS.items():
        cells = [method_name]
        for model_key in MODELS:
            fp, _ = fp_avgs[(model_key, method_file)]
            ksp, _ = ksp_avgs[(model_key, method_file)]
            cells += [f3(fp), f3(ksp)]
        print(" & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ---------------------------------------------------------------------------
# Section 4.2 - Empirical FDR Control
# ---------------------------------------------------------------------------

def table_fdr_power():
    """
    Combined table: average actual FDR and power at each target q, averaged across domains.
    Each method occupies two sub-rows: FDR and Power.
    """
    print("\n\n" + "=" * 100)
    print("TABLE: Empirical FDR Control and Power (Section 4.2)")
    print("  Average actual FDR and detection power across domains at each target q.")
    print("  FDR values <= q confirm FDR control.")
    print("=" * 100)

    def get_fdrs(result, method_file):
        section, fdr_key = METHOD_SECTIONS[method_file]
        return [
            None if result is None
            else result.get(section, {}).get(fdr_key, {}).get(q, {}).get("actual_fdr")
            for q in FDR_LEVELS
        ]

    def get_powers(result, method_file):
        section, fdr_key = METHOD_SECTIONS[method_file]
        return [
            None if result is None
            else result.get(section, {}).get(fdr_key, {}).get(q, {}).get("power")
            for q in FDR_LEVELS
        ]

    fdr_avgs   = collect_avg_multi(get_fdrs)
    power_avgs = collect_avg_multi(get_powers)

    col_w = 8
    model_items = list(MODELS.items())
    ql = [f"q={q}" for q in FDR_LEVELS]
    layers = [model_items[:2], model_items[2:]]

    # plain-text — one block per layer of 2 models
    for layer_models in layers:
        header = f"  {'Method':<16}" + "".join(
            "".join(f"{ql[j]:>{col_w}}" for j in range(len(FDR_LEVELS)))
            for _ in layer_models
        )
        subheader = f"  {'':<16}" + "".join(
            f"{m:>{col_w*len(FDR_LEVELS)}}" for _, m in layer_models
        )
        print(subheader)
        print(header)
        print("  " + "-" * (len(header) - 2))

        for method_file, method_name in METHODS.items():
            fdr_row   = f"  {method_name + ' FDR':<16}"
            power_row = f"  {'  Power':<16}"
            for model_key, _ in layer_models:
                fdr_vals, _   = fdr_avgs[(model_key, method_file)]
                power_vals, _ = power_avgs[(model_key, method_file)]
                if fdr_vals is None:
                    fdr_row   += "".join(f"{'---':>{col_w}}" for _ in FDR_LEVELS)
                    power_row += "".join(f"{'---':>{col_w}}" for _ in FDR_LEVELS)
                else:
                    for v in fdr_vals:
                        fdr_row += f"{f3(v):>{col_w}}"
                    for v in power_vals:
                        power_row += f"{f3(v):>{col_w}}"
            print(fdr_row)
            print(power_row)
        print()

    # LaTeX — two stacked layers, 2 models each
    print("\n\n--- LaTeX ---")
    n_q = len(FDR_LEVELS)

    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Empirical FDR control and detection power averaged across domains. "
          r"For each method, the first row reports the observed FDR and the second row "
          r"reports the detection power at the corresponding target level $q$.}")
    print(r"\label{tab:fdr_power}")

    for layer_idx, layer_models in enumerate(layers):
        n_layer = len(layer_models)
        col_spec = "ll" + (("c" * n_q + "|") * (n_layer - 1)) + "c" * n_q
        print(r"\begin{tabular}{" + col_spec + "}")
        print(r"\toprule")

        top = ["Method", ""] + [
            f"\\multicolumn{{{n_q}}}{{{'c|' if i < n_layer-1 else 'c'}}}{{{m}}}"
            for i, (_, m) in enumerate(layer_models)
        ]
        print(" & ".join(top) + r" \\")

        cmr_parts = []
        for i in range(n_layer):
            start = 3 + i * n_q
            cmr_parts.append(f"\\cmidrule(lr){{{start}-{start+n_q-1}}}")
        print("".join(cmr_parts))

        sub = ["", ""] + [f"$q={q}$" for q in FDR_LEVELS] * n_layer
        print(" & ".join(sub) + r" \\")
        print(r"\midrule")

        for method_file, method_name in METHODS.items():
            fdr_cells   = [r"\multirow{2}{*}{" + method_name + "}", "FDR"]
            power_cells = ["", "Power"]
            for model_key, _ in layer_models:
                fdr_vals, _   = fdr_avgs[(model_key, method_file)]
                power_vals, _ = power_avgs[(model_key, method_file)]
                if fdr_vals is None:
                    fdr_cells   += ["---"] * n_q
                    power_cells += ["---"] * n_q
                else:
                    fdr_cells   += [f3(v) for v in fdr_vals]
                    power_cells += [f3(v) for v in power_vals]
            print(" & ".join(fdr_cells)   + r" \\")
            print(" & ".join(power_cells) + r" \\")

        print(r"\bottomrule")
        print(r"\end{tabular}")
        if layer_idx < len(layers) - 1:
            print(r"\vspace{4pt}")
        print()

    print(r"\end{table*}")


if __name__ == "__main__":
    table_symmetry()
    table_fdr_power()
