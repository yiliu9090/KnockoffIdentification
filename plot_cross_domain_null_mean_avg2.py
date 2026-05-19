import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")

DOMAINS = [
    'AcademicResearch', 'ArtCulture', 'Business', 'EducationMaterial', 'Entertainment',
    'Environmental', 'Finance', 'FoodCusine', 'GovernmentPublic', 'LegalDocument',
    'MedicalText', 'NewsArticle', 'OnlineContent', 'PersonalCommunication', 'ProductReview',
    'Religious', 'Sports', 'TechnicalWriting', 'TravelTourism',
]
DOMAIN_LABELS = [
    'Academic', 'Art', 'Business', 'Education', 'Entertainment',
    'Environmental', 'Finance', 'Food', 'Government', 'Legal',
    'Medical', 'News', 'Online', 'Personal', 'Product',
    'Religious', 'Sports', 'Technical', 'Travel',
]
MODELS = {
    'GPT-3-Turbo':    'GPT-3.5T',
    'GPT-4o':         'GPT-4o',
    'Gemini-1.5-Pro': 'Gemini',
    'Llama-3-70B':    'Llama-70B',
}
METHODS = {
    'imbd':       'IMBD',
    'l2d':        'L2D',
    'likelihood': 'Likelihood',
}
FDR_LEVELS = ['0.1', '0.2', '0.3', '0.5']
MODEL_COLORS = ['#4878cf', '#6acc65', '#d65f5f', '#b47cc7']

model_keys  = list(MODELS.keys())
model_names = list(MODELS.values())
n_domains   = len(DOMAINS)
n_models    = len(model_keys)

bar_width = 0.18
x       = np.arange(n_domains)
offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_width

# Pre-load all result files
cache = {}
for method in METHODS:
    for model_key in model_keys:
        path = os.path.join(RESULT_DIR, f"cross_domain_null_mean_avg_{method}_{model_key}.json")
        if os.path.exists(path):
            with open(path) as f:
                cache[(method, model_key)] = json.load(f)

for q in FDR_LEVELS:
    fig, axes = plt.subplots(3, 2, figsize=(18, 10), sharey='col')

    for row, (method, method_name) in enumerate(METHODS.items()):
        fdr_matrix   = np.full((n_models, n_domains), np.nan)
        power_matrix = np.full((n_models, n_domains), np.nan)

        for mi, model_key in enumerate(model_keys):
            data = cache.get((method, model_key), {})
            for di, domain in enumerate(DOMAINS):
                entry = data.get(domain, {}).get(q, {})
                fdr_matrix[mi, di]   = entry.get('avg_fdr',   np.nan)
                power_matrix[mi, di] = entry.get('avg_power', np.nan)

        for ax, matrix, metric in zip(axes[row], [fdr_matrix, power_matrix], ['FDR', 'Power']):
            for mi, (name, color) in enumerate(zip(model_names, MODEL_COLORS)):
                ax.bar(x + offsets[mi], matrix[mi], width=bar_width,
                       label=name, color=color, alpha=0.85)

            if metric == 'FDR':
                ax.axhline(float(q), color='black', linewidth=1.0,
                           linestyle='--', label=f'q={q}')

            ax.set_xticks(x)
            ax.set_xticklabels(DOMAIN_LABELS, rotation=45, ha='right', fontsize=7.5)
            ax.set_ylabel(metric, fontsize=9)
            ax.set_title(f'{method_name} — {metric} at $q={q}$', fontsize=9)
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.set_xlim(-0.5, n_domains - 0.5)
            if row == 0 and metric == 'FDR':
                ax.legend(fontsize=8, ncol=2, loc='upper right')

    fig.suptitle(
        f'Cross-domain null mean transfer (avg over sources) — FDR and Power at $q={q}$',
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    tag = q.replace('.', '')
    out = os.path.join(os.path.dirname(__file__), f'cross_domain_null_mean_avg2_q{tag}.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')
