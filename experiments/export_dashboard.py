"""Tables for the results dashboard (results/dashboard/*.json), computed from results/ and data/ exactly as
balanced accuracy, two-level bootstrap (seeds, then rows) with seed 0 and 10,000 resamples for the deltas.

    python -m experiments.export_dashboard
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from adapt.problem import RULES
from adapt.prompts import LEVELS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'dashboard'
OUT.mkdir(parents=True, exist_ok=True)
SETS = ['test', 'near_threshold', 'd_pairs', 'k_pairs', 'g_pairs', 'unseen', 'longer_than_train']
rows = {s: {r['id']: r for r in map(json.loads, open(ROOT / 'data' / f'{s}.jsonl', encoding='utf-8'))} for s in SETS}
test = rows['test']
FOLDERS = {'qwen2.5-0.5b': ('Qwen2.5-0.5B', 1200), 'qwen2.5-1.5b': ('Qwen2.5-1.5B', 1200),
           'qwen2.5-3b': ('Qwen2.5-3B', 2400), 'llama-3.2-1b': ('Llama-3.2-1B', 2400),
           'llama-3.2-3b': ('Llama-3.2-3B', 2400), 'phi-3-mini': ('Phi-3-mini', 1200)}
CFGS = ['zero_shot', 'full', 'compressed', 'progressive_4_levels', 'progressive', 'full_code', 'compressed_mass_sentences',
        'progressive_mass_sentences', 'compressed_payload_rule', 'compressed_payload_rule_mass_sentences', 'full_4x_data',
        'full_2400_steps']
# controls not reported in the paper, kept in results/exploratory/<model>/, shown in the dashboard next to the others
EXPLORATORY = ['compressed_mass_sentences', 'progressive_mass_sentences', 'compressed_payload_rule_mass_sentences',
               'full_2400_steps']
r1 = lambda x: None if x is None else round(float(x), 1)


def load(folder):
    runs = defaultdict(list)
    for p in sorted((ROOT / 'results' / folder).glob('*.json')):
        d = json.loads(p.read_text(encoding='utf-8'))
        runs[re.sub(r'_s\d+$', '', d['name'])].append(d)
    return runs


runs = {f: load(f) for f in FOLDERS}
for f in FOLDERS:
    ex = load(f'exploratory/{f}')
    runs[f].update({c: ex[c] for c in EXPLORATORY if c in ex})

# ---- main --------------------------------------------------------------------------------------------------------------
main = []
for f, (model, steps) in FOLDERS.items():
    for c in CFGS:
        ds = runs[f].get(c)
        if not ds:
            continue
        m = lambda k, key='balanced': r1(np.mean([d['results'][k][key] for d in ds]))
        by_seed = {d['seed']: d['results']['test']['balanced'] for d in ds}
        main.append({'run': f'{model} {c}', 'model': model, 'config': c, 'level': ds[0]['final_level'],
                     'steps': 0 if c == 'zero_shot' else ds[0]['steps'], 'seeds': len(ds), 'test': m('test'),
                     'test_s3407': r1(by_seed.get(3407)), 'test_s1': r1(by_seed.get(1)), 'test_s2': r1(by_seed.get(2)),
                     'near_threshold': m('near_threshold'), 'd_pairs': m('d_pairs', 'pairs_both_right'),
                     'k_pairs': m('k_pairs', 'pairs_both_right'), 'g_pairs': m('g_pairs', 'pairs_both_right'),
                     'unseen': m('unseen'), 'longer_than_train': m('longer_than_train'),
                     'tokens': round(float(np.mean([d['results']['test']['tokens_mean'] for d in ds])))})

# ---- deltas (paired) ----------------------------------------------------------------------------------------------


def correct(d, name):
    return {i: pp == test[i]['A']['feasible'] for i, pp, _ in d['predictions'][name]}


def acc_vec(ds, name):
    cs = [correct(d, name) for d in ds]
    ids = sorted(cs[0])
    return ids, np.array([[c[i] for i in ids] for c in cs], float).mean(0)


def delta(da, na, db, nb):
    """Balanced-accuracy difference a - b on the same test rows (mean over seeds), with a two-level bootstrap interval
    (round 4): each resample draws the seeds of a and of b with replacement, then the rows, so the interval carries the
    spread between seeds as well as between rows. Seed 0, 10,000 resamples (2,000 left the bounds about 0.3 apart from run to run)."""
    ca, cb = [correct(d, na) for d in da], [correct(d, nb) for d in db]
    ids = sorted(ca[0])
    A = np.array([[c[i] for i in ids] for c in ca], float)
    B = np.array([[c[i] for i in ids] for c in cb], float)
    y = np.array([test[i]['A']['feasible'] for i in ids])

    def bal(ia, ib, idx):
        va, vb, yy = A[ia][:, idx].mean(0), B[ib][:, idx].mean(0), y[idx]
        return 100 * np.mean([np.mean(va[yy == k]) - np.mean(vb[yy == k]) for k in (True, False)])
    rng = np.random.default_rng(0)
    bs = [bal(rng.integers(0, len(A), len(A)), rng.integers(0, len(B), len(B)), rng.integers(0, len(ids), len(ids)))
          for _ in range(10000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    b = {d['seed']: d for d in db}
    seeds = {d['seed']: d['results'][na]['balanced'] - b[d['seed']]['results'][nb]['balanced'] for d in da if d['seed'] in b}
    return bal(np.arange(len(A)), np.arange(len(B)), np.arange(len(ids))), lo, hi, seeds


COMPS = []  # (group, folder, label, a, a-set, b, b-set, reading)
NI = 'non-inferior if the lower bound > -2'
for grp, label, a, na, b, nb, note in [
        ('Compressed', 'compressed - full', 'compressed', 'test', 'full', 'test', NI),
        ('Compressed', 'progressive_4_levels - full', 'progressive_4_levels', 'test', 'full', 'test', NI),
        ('Compressed', 'progressive - full', 'progressive', 'test', 'full', 'test', NI),
        ('Progressive against one-step compression', 'progressive_4_levels - compressed', 'progressive_4_levels', 'test', 'compressed',
         'test', 'better if the lower bound > 0'),
        ('Text cut without retraining', 'full@compressed - full', 'full', 'test@compressed', 'full', 'test', 'how much is lost')]:
    for f in FOLDERS:
        COMPS.append((grp, f, label, a, na, b, nb, note))
COMPS += [
    ('Control', 'qwen2.5-1.5b', 'compressed_mass_sentences - compressed (mass sentence added)',
     'compressed_mass_sentences', 'test', 'compressed', 'test', 'compressed_mass_sentences'),
    ('Control', 'qwen2.5-1.5b', 'compressed_payload_rule - compressed (payload rule sentence added)', 'compressed_payload_rule',
     'test', 'compressed', 'test', 'compressed_payload_rule'),
    ('Control', 'qwen2.5-1.5b', 'compressed_payload_rule_mass_sentences - compressed (both sentences added)',
     'compressed_payload_rule_mass_sentences', 'test', 'compressed', 'test', 'compressed_payload_rule_mass_sentences'),
    ('Control', 'qwen2.5-1.5b', 'compressed_payload_rule - full', 'compressed_payload_rule', 'test', 'full', 'test', NI),
    ('Control', 'phi-3-mini', 'full_code - full (narrowest clearance and battery need precomputed)', 'full_code', 'test', 'full', 'test',
     'full_code'),
    ('More data', 'phi-3-mini', 'full_4x_data - full (4x the training data)', 'full_4x_data', 'test', 'full', 'test',
     'H6'),
    ('More data', 'phi-3-mini', 'full_2400_steps - full (same data, 2,400 steps)', 'full_2400_steps', 'test',
     'full', 'test', 'H6'),
]
deltas = []
for group, f, label, a, na, b, nb, note in COMPS:
    d0, lo, hi, seeds = delta(runs[f][a], na, runs[f][b], nb)
    model = FOLDERS[f][0]
    deltas.append({'id': f'{model} | {label}', 'group': group, 'model': model, 'comparison': label, 'delta': r1(d0),
                   'lo': r1(lo), 'hi': r1(hi), 's3407': r1(seeds.get(3407)), 's1': r1(seeds.get(1)), 's2': r1(seeds.get(2)),
                   'reading': note})

# ---- skills: accuracy on test rows where exactly one rule fails, by that rule ---------------------------------------------
REASON = {'payload': 'look-up', 'reach_pick': 'comparison', 'reach_place': 'comparison', 'height': 'comparison',
          'clearance': 'minimum', 'battery': 'arithmetic'}
skills = []
for f, (model, _) in FOLDERS.items():
    for c in CFGS:
        if c == 'zero_shot' or c not in runs[f]:
            continue
        for k in RULES:
            ids = [i for i, r in test.items() if r['A']['violated'] == [k]]
            acc = np.mean([[correct(d, 'test')[i] for i in ids] for d in runs[f][c]])
            skills.append({'cell': f'{model} {c} {k}', 'model': model, 'config': c, 'rule': k, 'reasoning': REASON[k],
                           'n_rows': len(ids), 'acc': r1(100 * acc)})

# ---- the 1.5B look-up across prompts -----------------------------------------------------------------------------------
ids = [i for i, r in test.items() if r['A']['violated'] == ['payload']]
lookup = []
for setting, c, name, what in [
        ('full@full', 'full', 'test', 'full'),
        ('full@no_role_question', 'full', 'test@no_role_question', 'role sentences removed'),
        ('full@no_field_descriptions', 'full', 'test@no_field_descriptions', 'field descriptions removed'),
        ('full@compressed', 'full', 'test@compressed', 'rule sentences removed'),
        ('compressed', 'compressed', 'test', 'trained at compressed'),
        ('compressed_mass_sentences', 'compressed_mass_sentences', 'test', 'compressed + mass sentence'),
        ('compressed_payload_rule', 'compressed_payload_rule', 'test', 'compressed + payload rule sentence'),
        ('compressed_payload_rule_mass_sentences', 'compressed_payload_rule_mass_sentences', 'test',
         'compressed + both sentences')]:
    ds = runs['qwen2.5-1.5b'][c]
    acc = np.mean([[correct(d, name)[i] for i in ids] for d in ds])
    lookup.append({'setting': setting, 'prompt': what, 'acc': r1(100 * acc),
                   'tokens': round(float(np.mean([d['results'][name]['tokens_mean'] for d in ds])))})

# ---- latency at batch 1 on a T4 (the zero-shot runs) -----------------------------------------------------------------------------
latency = []
for f, (model, _) in FOLDERS.items():
    b = runs[f]['zero_shot'][0]['bench']
    for lv in LEVELS:
        latency.append({'point': f'{model} {lv}', 'model': model, 'level': lv, 'tokens': round(b[lv]['tokens_mean']),
                        'ms': r1(b[lv]['ms_mean'])})

# ---- non-LLM reference points (as data/REPORT.md) ------------------------------------------------------------------------
rep = (ROOT / 'data' / 'REPORT.md').read_text(encoding='utf-8')
names = {'majority': 'Always the majority answer', 'object only': 'Object name only', 'D only': 'D only',
         'G + D (K-blind)': 'Without K (G and D)', 'G + K + D': 'G, K, D (raw numbers)', 'slacks': 'Rule slacks given'}
baselines = []
for line in rep.split('\n'):
    cells = [c.strip() for c in line.strip('|').split('|')]
    if cells and cells[0] in names:
        v = lambda s: float(s.split('/')[-1].strip().replace(',', '.'))
        baselines.append({'baseline': names[cells[0]], 'test': float(cells[1].replace(',', '.')),
                          'near_threshold': float(cells[2].replace(',', '.')), 'k_pairs': v(cells[4]), 'g_pairs': v(cells[5])})

# ---- latency on CPU (H7): a Kaggle CPU session (all four) and the team's machine (0.5B, 1.5B) ----------------------------
NAMES = {'qwen2.5-0.5b': 'Qwen2.5-0.5B', 'qwen2.5-1.5b': 'Qwen2.5-1.5B', 'qwen2.5-3b': 'Qwen2.5-3B',
         'phi-3-mini': 'Phi-3-mini'}
cpu = []
for where, folder in (('Kaggle CPU', 'cpu_latency'), ('Authors machine', 'exploratory/cpu_latency_team')):
    mach = json.loads((ROOT / 'results' / folder / 'MACHINE.json').read_text(encoding='utf-8'))
    for k, model in NAMES.items():
        p = ROOT / 'results' / folder / f'{k}.json'
        if not p.exists():
            continue
        b = json.loads(p.read_text(encoding='utf-8'))['bench']
        for lv in LEVELS:
            cpu.append({'point': f'{where} {model} {lv}', 'machine': where, 'cpu': mach['cpu'], 'model': model, 'level': lv,
                        'tokens': round(b[lv]['tokens_mean']), 'ms': round(b[lv]['ms_mean'])})

for name, data in [('main', main), ('deltas', deltas), ('skills', skills), ('lookup', lookup), ('latency', latency),
                   ('baselines', baselines), ('cpu', cpu)]:
    (OUT / f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=0), encoding='utf-8')
    print(name, len(data), 'rows')
