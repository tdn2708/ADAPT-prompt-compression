"""Every number of the paper as a LaTeX key -> outputs/numbers.tex, so the text never holds a typed number.

    python -m experiments.paper_numbers        (after experiments.export_dashboard, report, prompt_tokens, audit)

In the paper, \\R{KEY} prints a number in Vietnamese style (decimal comma, plain hyphen for minus) and \\Rraw{KEY} the
same number with a decimal point (for TikZ). A key that does not exist prints in red, so it cannot go unnoticed.
Sources: results/dashboard/*.json (computed by experiments/export_dashboard.py from results/ and data/), data/REPORT.md,
data/PROMPTS.md, data/manifest*.json, results/cpu_latency/ and results/exploratory/cpu_latency_team/.
Keys: <model>|<configuration>|<value>, with the model and configuration names of results/ (e.g.
phi-3-mini|compressed_code|test); a comparison is 'a - b' and a configuration scored on another prompt level is
'configuration@level' (e.g. full@compressed: the Full model reading the compressed prompt).
"""

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / 'results' / 'dashboard'
SHORT = {'Qwen2.5-0.5B': 'qwen2.5-0.5b', 'Qwen2.5-1.5B': 'qwen2.5-1.5b', 'Qwen2.5-3B': 'qwen2.5-3b',
         'Llama-3.2-1B': 'llama-3.2-1b', 'Llama-3.2-3B': 'llama-3.2-3b', 'Phi-3-mini': 'phi-3-mini'}
Q7B, Q15, PHI, L3B, Q05 = 'qwen2.5-7b', 'qwen2.5-1.5b', 'phi-3-mini', 'llama-3.2-3b', 'qwen2.5-0.5b'
load = lambda n: json.loads((DASH / f'{n}.json').read_text(encoding='utf-8'))
K = {}


def put(key, value, digits=1):
    if value is None:
        return
    K[key] = (float(value), digits)


for r in load('main'):
    m = SHORT[r['model']]
    for f in ('test', 'near_threshold', 'd_pairs', 'k_pairs', 'g_pairs', 'unseen', 'longer_than_train', 'test_s3407',
              'test_s1', 'test_s2'):
        put(f"{m}|{r['config']}|{f}", r[f])
    put(f"{m}|{r['config']}|tokens", r['tokens'], 0)
    put(f"{m}|{r['config']}|seeds", r['seeds'], 0)
    put(f"{m}|{r['config']}|steps", r['steps'], 0)
for r in load('deltas'):
    m = SHORT[r['model']]
    label = r['comparison'].split(' (')[0]          # e.g. progressive - full, full@compressed - full
    for f in ('delta', 'lo', 'hi', 's3407', 's1', 's2'):
        put(f'{m}|{label}|{f}', r[f])
for r in load('skills'):
    put(f"{SHORT[r['model']]}|{r['config']}|{r['rule']}", r['acc'])
for r in load('lookup'):
    put(f"lookup|{r['setting']}", r['acc'])
    put(f"lookup|{r['setting']}|tokens", r['tokens'], 0)
lat = load('latency')
for r in lat:
    m = SHORT[r['model']]
    put(f"{m}|T4|{r['level']}|ms", r['ms'])
    put(f"{m}|T4|{r['level']}|s", r['ms'] / 1000, 2)
    put(f"{m}|T4|{r['level']}|tokens", r['tokens'], 0)
for m in {SHORT[r['model']] for r in lat}:
    a, b = K[f'{m}|T4|full|ms'][0], K[f'{m}|T4|compressed|ms'][0]
    put(f'{m}|T4|cut', 100 * (b / a - 1), 0)
for r in load('cpu'):
    where = 'KCPU' if r['machine'].startswith('Kaggle') else 'LCPU'
    put(f"{SHORT[r['model']]}|{where}|{r['level']}|ms", r['ms'], 0)
    put(f"{SHORT[r['model']]}|{where}|{r['level']}|s", r['ms'] / 1000, 1)
for where in ('KCPU', 'LCPU'):
    for m in SHORT.values():
        if f'{m}|{where}|full|ms' in K:
            a, b = K[f'{m}|{where}|full|ms'][0], K[f'{m}|{where}|compressed|ms'][0]
            put(f'{m}|{where}|cut', 100 * (b / a - 1), 0)
for r in load('baselines'):
    b = {'Always the majority answer': 'maj', 'Object name only': 'obj', 'D only': 'donly', 'Without K (G and D)': 'kblind',
         'G, K, D (raw numbers)': 'gkd', 'Rule slacks given': 'slack'}[r['baseline']]
    for f in ('test', 'near_threshold', 'k_pairs', 'g_pairs'):
        put(f'base|{b}|{f}', r[f])

# ---- dataset (data/REPORT.md, data/manifest*.json, data/PROMPTS.md, data/AUDIT*.md) ----------------------------------------
man = json.loads((ROOT / 'data' / 'manifest.json').read_text(encoding='utf-8'))
for f, n in man['files'].items():
    put(f"data|{f.replace('.jsonl', '')}|rows", n, 0)
put('data|total', man['total_rows'], 0)
put('data|pairs|d_pairs', man['files']['d_pairs.jsonl'] / 2, 0)
put('data|pairs|k_pairs', man['files']['k_pairs.jsonl'] / 2, 0)
put('data|pairs|g_pairs', man['files']['g_pairs.jsonl'] / 2, 0)
x4 = json.loads((ROOT / 'data' / 'manifest_train_x4.json').read_text(encoding='utf-8'))
put('data|train_x4|rows', x4['rows'], 0)
rep = (ROOT / 'data' / 'REPORT.md').read_text(encoding='utf-8')
for line in rep.split('\n'):
    m_ = re.match(r'\| `(\w+)\.jsonl` \| (\d+) \| ([\d,.]+)% \| ([\d,.]+)% \| ([\d,.]+)% \|', line)
    if m_:
        f = m_.group(1)
        put(f'data|{f}|near', float(m_.group(4).replace(',', '.')))
        put(f'data|{f}|kdep', float(m_.group(5).replace(',', '.')))
pr = (ROOT / 'data' / 'PROMPTS.md').read_text(encoding='utf-8')
for line in pr.split('\n'):
    m_ = re.match(r'\| (full|no_role_question|no_field_descriptions|compressed) \| [^|]+ \| (\d+)[^|]*\| (\d+)[^|]*\| '
                  r'(\d+)[^|]*\| (\d+)[^|]*\|', line)
    if m_:
        put(f'tok|Qwen|{m_.group(1)}', int(m_.group(2)), 0)
        put(f'tok|Phi|{m_.group(1)}', int(m_.group(5)), 0)
for fam in ('Qwen', 'Phi'):
    put(f'tok|{fam}|cut', 100 * (K[f'tok|{fam}|compressed'][0] / K[f'tok|{fam}|full'][0] - 1), 0)
for name, fname in (('audit', 'AUDIT.md'), ('auditx4', 'AUDIT_X4.md')):
    a = re.search(r'\*\*(\d+)/(\d+) checks passed', (ROOT / 'data' / fname).read_text(encoding='utf-8'))
    put(f'{name}|pass', int(a.group(1)), 0)
    put(f'{name}|all', int(a.group(2)), 0)


# the example row of Fig. 1 (test/726): its own token counts, Qwen2.5 tokenizer from the local cache
try:
    from transformers import AutoTokenizer
    from adapt.prompts import LEVELS, Prompter
    ex = next(json.loads(l) for l in open(ROOT / 'data' / 'test.jsonl', encoding='utf-8') if '"test/726"' in l)
    P = Prompter(AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'))
    for lv in LEVELS:
        put(f'ex|tokens|{lv}', P.n_tokens(ex, lv), 0)
    _test = [json.loads(l) for l in open(ROOT / 'data' / 'test.jsonl', encoding='utf-8')]
    for lv in ('full_code', 'compressed_code'):          # the prompts of Full + code and of the proposed system
        put(f'tok|Qwen|{lv}', sum(P.n_tokens(r, lv) for r in _test) / len(_test), 0)
    put('ex|tokens|compressed_code', P.n_tokens(ex, 'compressed_code'), 0)
    # Fig. 4 bars: tokens of each source of the example prompt, e.g. ex|seg|full|D (role = the first I lines,
    # question = the last I lines, code = the precomputed lines, template = chat template and answer prefix)
    for lv in LEVELS + ['compressed_code']:
        seen = set()
        for tag, n in P.segment_tokens(ex, lv):
            name = {'C': 'code', 'T': 'template'}.get(tag, tag)
            if tag == 'I':
                name = 'question' if 'role' in seen else 'role'
            seen.add(name)
            put(f'ex|seg|{lv}|{name}', n, 0)
except Exception as e:                                   # no tokenizer offline: the figure shows red keys
    print('example tokens skipped:', type(e).__name__)

# the problem itself (adapt/problem.py) and the hand check (data/HAND_CHECK.md: human first, then the language model)
from adapt import problem as pb
for name, n in (('templates', len(pb.TEMPLATES)), ('templates_ho', len(pb.HELD_OUT_TEMPLATES)), ('objects', len(pb.OBJECTS)),
                ('objects_ho', len(pb.HELD_OUT_OBJECTS)), ('places', len(pb.PLACES)), ('places_ho', len(pb.HELD_OUT_PLACES))):
    put(f'prob|{name}', n, 0)
for k, v in pb.DEFAULT.items():
    put(f'prob|k0|{k}', v, pb.DIGITS[k] if pb.DIGITS[k] == 0 else len(str(v).split('.')[1]))
# the truncated normal of the limits: its parameters, the share within one sigma, and the payload limit as the example
lo, hi = pb.FACTOR
put('prob|sigma', pb.SIGMA, 1)
put('prob|lo', lo, 1)
put('prob|hi', hi, 1)
cdf = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
put('prob|in1sd', 100 * (cdf(1) - cdf(-1)) / (cdf((hi - 1) / pb.SIGMA) - cdf((lo - 1) / pb.SIGMA)), 0)
p0 = pb.DEFAULT['payload']
for name, x in (('sd', pb.SIGMA * p0), ('lo', lo * p0), ('hi', hi * p0), ('minus', (1 - pb.SIGMA) * p0),
                ('plus', (1 + pb.SIGMA) * p0)):
    put(f'prob|payload|{name}', x, 2)
hand = re.findall(r'(\d+)/(\d+) match', (ROOT / 'data' / 'HAND_CHECK.md').read_text(encoding='utf-8'))
for who, (a, b) in zip(('human', 'llm'), hand):
    put(f'hand|{who}|ok', int(a), 0)
    put(f'hand|{who}|all', int(b), 0)

for m in SHORT.values():
    run = json.loads((ROOT / 'results' / m / 'full_s3407.json').read_text(encoding='utf-8'))
    put(f'{m}|lora|M', run['lora_trainable_params'] / 1e6, 1)

# ranges over the six models, so the text can say "from x to y" without a typed number
MODELS6 = list(SHORT.values())
for rule in ('payload', 'reach_pick', 'reach_place', 'height', 'clearance', 'battery'):
    vals = [K[f'{m}|full|{rule}'][0] for m in MODELS6]
    put(f'full|{rule}|min', min(vals))
    put(f'full|{rule}|max', max(vals))
for f in ('test',):
    vals = [K[f'{m}|zero_shot|{f}'][0] for m in MODELS6]
    put(f'zero_shot|{f}|min', min(vals))
    put(f'zero_shot|{f}|max', max(vals))
vals = [K[f'{m}|full@compressed - full|delta'][0] for m in MODELS6]
put('full@compressed|delta|min', min(vals))
put('full@compressed|delta|max', max(vals))
# the other progressive schedule progressive_4_levels (a quarter of the steps at each level of Fig. 3), tried besides
# progressive: its range against Full, and on how many of the six models it is non-inferior (lower bound > -2)
vals = [K[f'{m}|progressive_4_levels - full|delta'][0] for m in MODELS6]
put('progressive_4_levels|delta|min', min(vals))
put('progressive_4_levels|delta|max', max(vals))
put('progressive_4_levels|ni|n', sum(K[f'{m}|progressive_4_levels - full|lo'][0] > -2 for m in MODELS6), 0)
for where, ms in (('T4', ['qwen2.5-1.5b', 'qwen2.5-3b', 'llama-3.2-1b', 'llama-3.2-3b', 'phi-3-mini']),
                  ('KCPU', ['qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-3b', 'phi-3-mini'])):
    vals = [abs(K[f'{m}|{where}|cut'][0]) for m in ms]
    put(f'{where}|cut|min', min(vals), 0)
    put(f'{where}|cut|max', max(vals), 0)

put(f'{Q15}|compressed_payload_rule|extra', K[f'{Q15}|compressed_payload_rule|tokens'][0]
    - K[f'{Q15}|compressed|tokens'][0], 0)                          # tokens of the one kept payload sentence

# ---- round 3: straight from the run files, every prediction matched to data/ by id ------------------
import glob as _glob
import numpy as _np

DATA = {s: {json.loads(l)['id']: json.loads(l) for l in open(ROOT / 'data' / f'{s}.jsonl', encoding='utf-8')}
        for s in ('test', 'near_threshold', 'd_pairs', 'k_pairs', 'g_pairs', 'unseen', 'longer_than_train')}
_y = lambda r: bool(r['A']['feasible'])


def _files(pat):
    return [json.loads(Path(f).read_text(encoding='utf-8')) for f in sorted(_glob.glob(str(ROOT / 'results' / pat)))]


def _bal(rows, pred):
    y = _np.array([_y(r) for r in rows]); p = _np.array(pred)
    return 100 * float(_np.mean([_np.mean(p[y == k] == k) for k in (True, False)]))


def summarize(m, c, pat, longer_pat=None):
    """Mean over the seeds of a configuration: every set, every single-violation rule, the false-feasible rate on test
    (infeasible rows answered feasible), time per decision and new tokens (step-by-step runs)."""
    ds = _files(pat)
    if not ds:
        return
    put(f'{m}|{c}|seeds', len(ds), 0)
    for s, key in (('test', 'test'), ('near_threshold', 'near_threshold'), ('unseen', 'unseen'), ('longer_than_train', 'longer_than_train'),
                   ('d_pairs', 'd_pairs'), ('k_pairs', 'k_pairs'), ('g_pairs', 'g_pairs')):
        src = _files(longer_pat) if (s == 'longer_than_train' and longer_pat) else ds
        if not src or not all(s in d['predictions'] for d in src):
            continue
        vals = []
        for d in src:
            pred = {i: bool(p) for i, p, _ in d['predictions'][s]}
            assert set(pred) == set(DATA[s]), (pat, s)
            if key.endswith('pairs'):
                ok = defaultdict(list)
                for i, p in pred.items():
                    ok[DATA[s][i]['pair_id']].append(p == _y(DATA[s][i]))
                vals.append(100 * float(_np.mean([all(v) for v in ok.values()])))
            else:
                vals.append(_bal([DATA[s][i] for i in pred], list(pred.values())))
        put(f'{m}|{c}|{key}', _np.mean(vals))
    acc, ff = defaultdict(list), []
    for d in ds:
        for i, p, _ in d['predictions']['test']:
            r = DATA['test'][i]
            if r['n_violated'] == 1:
                acc[r['A']['violated'][0]].append(bool(p) == _y(r))
            if not _y(r):
                ff.append(bool(p))
    for rule, v in acc.items():
        put(f'{m}|{c}|{rule}', 100 * _np.mean(v))
    put(f'{m}|{c}|ff', 100 * _np.mean(ff))
    b = [list(d['bench'].values())[0] for d in ds if d.get('bench') and 'ms_mean' in list(d['bench'].values())[0]]
    if b and 'new_tokens_mean' in b[0]:
        put(f'{m}|{c}|ms', _np.mean([x['ms_mean'] for x in b]), 0)
        put(f'{m}|{c}|s', _np.mean([x['ms_mean'] for x in b]) / 1000, 1)
        put(f'{m}|{c}|newtok', _np.mean([x['new_tokens_mean'] for x in b]), 0)
    for d, seed in zip(ds, [x['seed'] for x in ds]):
        sel = {i: bool(p) for i, p, _ in d['predictions']['test']}
        put(f'{m}|{c}|test_s{seed}', _bal([DATA['test'][i] for i in sel], list(sel.values())))
        bb = list((d.get('bench') or {}).values())
        if bb and 'new_tokens_mean' in bb[0]:
            put(f'{m}|{c}|s_s{seed}', bb[0]['ms_mean'] / 1000, 1)
        acc1 = defaultdict(list)
        for i, p in sel.items():
            r = DATA['test'][i]
            if r['n_violated'] == 1:
                acc1[r['A']['violated'][0]].append(p == _y(r))
        for rule in ('clearance', 'battery'):
            put(f'{m}|{c}|{rule}_s{seed}', 100 * _np.mean(acc1[rule]))


from collections import defaultdict  # noqa: E402

# the false-feasible rate of the one-token configurations of the first two rounds
for m in MODELS6:
    for c in ('full', 'compressed', 'progressive_4_levels', 'progressive', 'full_code'):
        ff = [bool(p) for d in _files(f'{m}/{c}_s*.json') for i, p, _ in d['predictions']['test']
              if not _y(DATA['test'][i])]
        if ff:
            put(f'{m}|{c}|ff', 100 * _np.mean(ff))
EX = 'exploratory/'
for m, c, pat, longer in (
        (Q15, 'step_by_step', f'{Q15}/step_by_step_s*.json', f'{EX}{Q15}/step_by_step_longer_eval_s*.json'),
        (Q15, 'step_by_step_compressed', f'{EX}{Q15}/step_by_step_compressed_s*.json',
         f'{EX}{Q15}/step_by_step_compressed_longer_eval_s*.json'),
        (PHI, 'step_by_step', f'{PHI}/step_by_step_s*.json', f'{EX}{PHI}/step_by_step_longer_eval_s*.json'),
        (PHI, 'step_by_step_compressed', f'{EX}{PHI}/step_by_step_compressed_s*.json',
         f'{EX}{PHI}/step_by_step_compressed_longer_eval_s*.json'),
        (L3B, 'step_by_step', f'{L3B}/step_by_step_s*.json', None),
        (L3B, 'step_by_step_compressed', f'{EX}{L3B}/step_by_step_compressed_s*.json', None),
        (Q7B, 'full', f'{Q7B}/full_s*.json', None), (Q7B, 'compressed', f'{Q7B}/compressed_s*.json', None),
        ('qwen2.5-3b', 'compressed_grad_clip', f'{EX}qwen2.5-3b/compressed_grad_clip_s*.json', None),
        (PHI, 'compressed_code', f'{PHI}/compressed_code_s*.json', None),
        (L3B, 'compressed_code', f'{L3B}/compressed_code_s*.json', None),
        (PHI, 'full_rerun', f'{PHI}/full_rerun_s*.json', None),
        (Q15, 'compressed_code', f'{Q15}/compressed_code_s*.json', None),
        (Q05, 'compressed_code', f'{Q05}/compressed_code_s*.json', None),
        (PHI, 'compressed_code_2400_steps', f'{PHI}/compressed_code_2400_steps_s*.json', None),
        (PHI, 'full_code', f'{PHI}/full_code_s*.json', None)):
    summarize(m, c, pat, longer)
# ranges over the three step-by-step models, and the 7B compressed against its full prompt (rows averaged over the seeds
# first, then a paired bootstrap over the test rows)
SBS = 'step_by_step'
for rule in ('clearance', 'battery'):
    vals = [K[f'{m}|{SBS}|{rule}'][0] for m in (Q15, PHI, L3B) if f'{m}|{SBS}|{rule}' in K]
    put(f'{SBS}|{rule}|min', min(vals)); put(f'{SBS}|{rule}|max', max(vals))
vals = [K[f'{m}|{SBS}|ms'][0] / K[f'{m}|T4|full|ms'][0] for m in (Q15, PHI, L3B) if f'{m}|{SBS}|ms' in K]
put(f'{SBS}|slow|min', min(vals), 0); put(f'{SBS}|slow|max', max(vals), 0)
vals = [K[f'{m}|{SBS}|s'][0] for m in (Q15, PHI, L3B) if f'{m}|{SBS}|s' in K]
put(f'{SBS}|s|min', min(vals)); put(f'{SBS}|s|max', max(vals))


# ---- the capability map, cell by cell (round 4): rows per rule, a 95% interval for every cell (two-level bootstrap: the
# seeds, then the rows of that rule), and the false-infeasible rate (feasible test rows answered infeasible) of each row
MAP = [(m, 'full', f'{m}/full_s*.json') for m in MODELS6] + [
       (Q7B, 'full', f'{Q7B}/full_s*.json'), (PHI, 'full_4x_data', f'{PHI}/full_4x_data_s*.json'),
       (PHI, 'full_code', f'{PHI}/full_code_s*.json'),
       (Q15, 'step_by_step', f'{Q15}/step_by_step_s*.json'), (L3B, 'step_by_step', f'{L3B}/step_by_step_s*.json'),
       (PHI, 'step_by_step', f'{PHI}/step_by_step_s*.json'),
       (PHI, 'compressed_code', f'{PHI}/compressed_code_s*.json'), (L3B, 'compressed_code', f'{L3B}/compressed_code_s*.json'),
       (Q15, 'compressed', f'{Q15}/compressed_s*.json'), (Q15, 'compressed_code', f'{Q15}/compressed_code_s*.json'),
       (Q05, 'compressed_code', f'{Q05}/compressed_code_s*.json')]
_rule_rows = {rule: [i for i, r in DATA['test'].items() if r['A']['violated'] == [rule]] for rule in pb.RULES}
for rule, ids in _rule_rows.items():
    put(f'n|{rule}', len(ids), 0)
_feas = [i for i, r in DATA['test'].items() if _y(r)]
_half = []
for m, c, pat in MAP:
    ds = _files(pat)
    if not ds:
        continue
    preds = [{i: bool(p) for i, p, _ in d['predictions']['test']} for d in ds]
    put(f'{m}|{c}|fa', 100 * _np.mean([not p[i] for p in preds for i in _feas]))
    if len(preds) > 1:                                   # spread over the seeds of the same row (sample standard deviation)
        put(f'{m}|{c}|fa|sd', _np.std([100 * _np.mean([not p[i] for i in _feas]) for p in preds], ddof=1))
    rng = _np.random.default_rng(0)
    for rule, ids in _rule_rows.items():
        M = _np.array([[float(not p[i]) for i in ids] for p in preds])          # violation found
        if len(M) > 1:
            put(f'{m}|{c}|{rule}|sd', _np.std(100 * M.mean(axis=1), ddof=1))
        boot = [M[rng.integers(0, len(M), len(M))][:, rng.integers(0, len(ids), len(ids))].mean() for _ in range(10000)]
        lo, hi = _np.percentile(boot, [2.5, 97.5])
        # the bootstrap collapses near 0% and 100%: keep the wider of it and the Wilson interval on the rows of the rule
        p_, n_, z = M.mean(), len(ids), 1.96
        mid, w = (p_ + z * z / (2 * n_)) / (1 + z * z / n_), z * _np.sqrt(p_ * (1 - p_) / n_ + z * z / (4 * n_ * n_)) / (1 + z * z / n_)
        lo, hi = min(lo, mid - w), max(hi, mid + w)
        put(f'{m}|{c}|{rule}|lo', 100 * lo, 0); put(f'{m}|{c}|{rule}|hi', 100 * hi, 0)
        _half.append((100 * (hi - lo) / 2, len(M)))
for k in (1, 2, 3):
    w = [h for h, s in _half if s == k]
    if w:
        put(f'mapci|half|{k}seed', max(w), 0)


def _paired(pat_a, pat_b, n_boot=10000, set_a='test', set_b='test', ids=None):
    """a - b in balanced accuracy on the same rows (mean over seeds), two-level bootstrap as export_dashboard.delta: the
    seeds of a and of b, then the rows. set_a / set_b: result sets; ids: rows of set_a, with ids_b = their counterparts
    in set_b (free-form rows against their source test rows)."""
    ids = ids or sorted(DATA['test'])
    pairs = ids if isinstance(ids[0], tuple) else [(i, i) for i in ids]
    def mat(pat, name, k):
        out = []
        for d in _files(pat):
            p = {i: bool(x) for i, x, _ in d['predictions'][name]}
            out.append([float(p[pr[k]] == _y(DATA['test'][pairs[j][1]])) for j, pr in enumerate(pairs)])
        return _np.array(out)
    A, B = mat(pat_a, set_a, 0), mat(pat_b, set_b, 1)
    y = _np.array([_y(DATA['test'][j]) for _, j in pairs])
    bal = lambda M, ia, idx: 100 * _np.mean([M[ia][:, idx].mean(0)[y[idx] == k].mean() for k in (True, False)])
    rng = _np.random.default_rng(0)
    boot = []
    for _ in range(n_boot):
        ia, ib, s = rng.integers(0, len(A), len(A)), rng.integers(0, len(B), len(B)), rng.integers(0, len(y), len(y))
        boot.append(bal(A, ia, s) - bal(B, ib, s))
    full = _np.arange(len(y))
    return bal(A, _np.arange(len(A)), full) - bal(B, _np.arange(len(B)), full), *_np.percentile(boot, [2.5, 97.5])


if _files(f'{Q7B}/compressed_s*.json'):
    d0, lo, hi = _paired(f'{Q7B}/compressed_s*.json', f'{Q7B}/full_s*.json')
    put(f'{Q7B}|compressed - full|delta', d0); put(f'{Q7B}|compressed - full|lo', lo)
    put(f'{Q7B}|compressed - full|hi', hi)
vals = [d['results']['test@compressed']['balanced'] for d in _files(f'{Q7B}/full_s*.json')]
put(f'{Q7B}|full@compressed|test', _np.mean(vals))

# ---- round 4: the proposed system, the free-form requests, LLMLingua-2, the 7B zero-shot
# and the extraction baseline (the LM names the object, code checks the six rules)
FF = {json.loads(l)['id']: json.loads(l) for l in open(ROOT / 'data' / 'free_form.jsonl', encoding='utf-8')}
put('data|free_form|rows', len(FF), 0)
put('data|free_form|distractor', sum(bool(r['G'].get('distractor')) for r in FF.values()), 0)
_ffpairs = [(i, FF[i]['source']) for i in sorted(FF)]
# Phi-3-mini: against Full trained again in the same batch of runs (full_rerun); Llama-3.2-3B: against its Full runs
for m, ref_name in ((PHI, 'full_rerun'), (L3B, 'full')):
    pat, ref = f'{m}/compressed_code_s*.json', f'{m}/{ref_name}_s*.json'
    d0, lo, hi = _paired(pat, ref)
    k = f'{m}|compressed_code - {ref_name}'
    put(f'{k}|delta', d0); put(f'{k}|lo', lo); put(f'{k}|hi', hi)
    ratio = [d['bench']['compressed_code']['ms_mean'] / d['bench']['full']['ms_mean'] for d in _files(pat)
             if d.get('bench')]
    put(f'{m}|compressed_code|ratio', _np.mean(ratio), 2)
    put(f'{m}|T4|compressed_code|ms', _np.mean([d['bench']['compressed_code']['ms_mean'] for d in _files(pat)
                                                 if d.get('bench')]), 0)
for m, c in ((PHI, 'full_rerun'), (PHI, 'compressed_code'), (L3B, 'compressed_code')):
    pat = f'{m}/{c}_s*.json'
    put(f'{m}|{c}|free_form', _np.mean([d['results']['free_form']['balanced'] for d in _files(pat)]))
    d0, lo, hi = _paired(pat, pat, set_a='free_form', ids=_ffpairs)
    put(f'{m}|{c}|free_form - test|delta', d0); put(f'{m}|{c}|free_form - test|lo', lo)
    put(f'{m}|{c}|free_form - test|hi', hi)
d0, lo, hi = _paired(f'{PHI}/compressed_code_s*.json', f'{PHI}/progressive_s*.json')
put(f'{PHI}|compressed_code - progressive|delta', d0); put(f'{PHI}|compressed_code - progressive|lo', lo)
put(f'{PHI}|compressed_code - progressive|hi', hi)


def _seed_deltas(key, pat_a, pat_b):
    """test balanced accuracy of a minus b for every seed run in both (same seed, same rows): key|s<seed> (Fig. 5 dots)."""
    b = {d['seed']: d['results']['test']['balanced'] for d in _files(pat_b)}
    for d in _files(pat_a):
        if d['seed'] in b:
            put(f"{key}|s{d['seed']}", d['results']['test']['balanced'] - b[d['seed']])


_seed_deltas(f'{L3B}|compressed - full', f'{L3B}/compressed_s*.json', f'{L3B}/full_s*.json')   # adds seeds 4 and 5
_seed_deltas(f'{PHI}|compressed_code - full_rerun', f'{PHI}/compressed_code_s*.json', f'{PHI}/full_rerun_s*.json')
_seed_deltas(f'{L3B}|compressed_code - full', f'{L3B}/compressed_code_s*.json', f'{L3B}/full_s*.json')
LING = {i: r for i, r in ((json.loads(l)['id'], json.loads(l)) for l in open(ROOT / 'data' / 'llmlingua2.jsonl', encoding='utf-8'))
        if i.startswith('test/')}
_kept = [i for i, r in LING.items() if r['numbers_kept'] == r['numbers']]
put('lingua|kept', len(_kept), 0); put('lingua|kept|pct', 100 * len(_kept) / len(LING), 0)
put('lingua|lost|pct', 100 - 100 * len(_kept) / len(LING), 0)
put('lingua|tokens', _np.mean([r['tokens'] for r in LING.values()]), 0)
_p1 = _files(f'{PHI}/full_rerun_s*.json')
RR = f'{PHI}|full_rerun'
put(f'{RR}|test@llmlingua2', _np.mean([d['results']['test@llmlingua2']['balanced'] for d in _p1]))
put(f'{RR}|test@llmlingua2|predtrue', _np.mean([d['results']['test@llmlingua2']['pred_true_rate'] for d in _p1]))
put(f'{RR}|test@compressed', _np.mean([d['results']['test@compressed']['balanced'] for d in _p1]))
for name, ids in (('kept', _kept), ('lost', [i for i in LING if i not in set(_kept)])):
    put(f'{RR}|test@llmlingua2|{name}',
        _np.mean([_bal([DATA['test'][i] for i in ids],
                       [{j: bool(x) for j, x, _ in d['predictions']['test@llmlingua2']}[i] for i in ids]) for d in _p1]))
d0, lo, hi = _paired(f'{PHI}/full_rerun_s*.json', f'{PHI}/compressed_s*.json', set_a='test@llmlingua2')
k = f'{PHI}|full_rerun@llmlingua2 - compressed'
put(f'{k}|delta', d0); put(f'{k}|lo', lo); put(f'{k}|hi', hi)
_pl = _files(f'{PHI}/llmlingua2_s*.json')     # H13b: trained and scored on the LLMLingua-2 prompts
if _pl:
    LL = f'{PHI}|llmlingua2'
    put(f'{LL}|test', _np.mean([d['results']['test']['balanced'] for d in _pl]))
    put(f'{LL}|seeds', len(_pl), 0)
    d0, lo, hi = _paired(f'{PHI}/llmlingua2_s*.json', f'{PHI}/compressed_s*.json')
    put(f'{LL} - compressed|delta', d0); put(f'{LL} - compressed|lo', lo); put(f'{LL} - compressed|hi', hi)
    for name, ids in (('kept', _kept), ('lost', [i for i in LING if i not in set(_kept)])):
        put(f'{LL}|{name}', _np.mean([_bal([DATA['test'][i] for i in ids],
                                            [{j: bool(x) for j, x, _ in d['predictions']['test']}[i] for i in ids])
                                       for d in _pl]))
    put(f'{LL}|ff', 100 * _np.mean([bool(x) for d in _pl for i, x, _ in d['predictions']['test']
                                    if not _y(DATA['test'][i])]))
# time per decision of the remaining points of the accuracy-time figure (T4, batch 1, the run that timed the levels)
for key, pat, lv in ((f'{PHI}|T4|llmlingua2|ms', f'{PHI}/llmlingua2_s*.json', 'llmlingua2'),
                     (f'{Q15}|T4|compressed_code|ms', f'{Q15}/compressed_code_s*.json', 'compressed_code'),
                     (f'{Q05}|T4|compressed_code|ms', f'{Q05}/compressed_code_s*.json', 'compressed_code')):
    ms = [d['bench'][lv]['ms_mean'] for d in _files(pat) if d.get('bench') and lv in d['bench']]
    if ms:
        put(key, _np.mean(ms), 0)
for c, f in (('zero_shot', f'{EX}{Q7B}/zero_shot'), ('zero_shot_step_by_step', f'{Q7B}/zero_shot_step_by_step')):
    d = _files(f'{f}.json')[0]
    put(f'{Q7B}|{c}|test', d['results']['test']['balanced'])
    put(f'{Q7B}|{c}|predtrue', d['results']['test']['pred_true_rate'])
    acc = defaultdict(list)
    for i, p, _ in d['predictions']['test']:
        r = DATA['test'][i]
        if r['n_violated'] == 1:
            acc[r['A']['violated'][0]].append(bool(p) == _y(r))
    for rule, v in acc.items():
        put(f'{Q7B}|{c}|{rule}', 100 * _np.mean(v))
d = _files(f'{Q7B}/zero_shot_step_by_step.json')[0]
ZSSB = f'{Q7B}|zero_shot_step_by_step'
put(f'{ZSSB}|unparsed', d['results']['test']['unparsed'], 0)
put(f'{ZSSB}|trunc', sum('Feasible:' not in g for _, g in d['generations']['test']), 0)
put(f'{ZSSB}|s', d['bench']['full']['ms_mean'] / 1000, 1)
for m in (Q7B, Q15):
    d = _files(f'{m}/object_naming.json')[0]
    for s in ('test', 'free_form'):
        put(f'object_naming|{m}|{s}', d['results'][s]['balanced'])
        put(f'object_naming|{m}|{s}|obj', d['results'][s]['object_acc'])
        put(f'object_naming|{m}|{s}|unparsed', d['results'][s]['unparsed'], 0)

# spread over the seeds: a |sd key (sample standard deviation) next to every mean that the result tables print; the mean of
# the same files must match the key already there, so the two cannot come from different runs
_SETS = (('test', 'test', 'balanced'), ('near_threshold', 'near_threshold', 'balanced'), ('d_pairs', 'd_pairs', 'pairs_both_right'),
         ('k_pairs', 'k_pairs', 'pairs_both_right'), ('g_pairs', 'g_pairs', 'pairs_both_right'),
         ('unseen', 'unseen', 'balanced'), ('longer_than_train', 'longer_than_train', 'balanced'))
_SPREAD = [(m, c, f'{m}/{c}_s*.json') for m in MODELS6
           for c in ('full', 'compressed', 'progressive_4_levels', 'progressive', 'full_code')]
_SPREAD += [(m, c, f'{m}/{c}_s*.json') for m, c in (
    (PHI, 'compressed_code'), (L3B, 'compressed_code'), (Q15, 'compressed_code'), (Q05, 'compressed_code'),
    (PHI, 'full_rerun'), (PHI, 'compressed_code_2400_steps'), (Q7B, 'full'), (Q7B, 'compressed'),
    (PHI, 'llmlingua2'), (Q15, 'compressed_payload_rule'), (PHI, 'full_4x_data'), (Q15, 'step_by_step'),
    (L3B, 'step_by_step'), (PHI, 'step_by_step'))]
for m, c, pat in _SPREAD:
    ds = _files(pat)
    if len(ds) < 2:
        continue
    for s, key, f in _SETS:
        if c == 'step_by_step' and s != 'test':          # its longer_than_train comes from a separate rerun (360 tokens)
            continue
        if not all(s in d['results'] and f in d['results'][s] for d in ds):
            continue
        v = [d['results'][s][f] for d in ds]
        k = f'{m}|{c}|{key}'
        if k in K:
            assert abs(_np.mean(v) - K[k][0]) < 0.06, (k, _np.mean(v), K[k][0])
        put(k + '|sd', _np.std(v, ddof=1))
    ff = [100 * _np.mean([bool(p) for i, p, _ in d['predictions']['test'] if not _y(DATA['test'][i])]) for d in ds]
    if f'{m}|{c}|ff' in K:
        assert abs(_np.mean(ff) - K[f'{m}|{c}|ff'][0]) < 0.06, (m, c, 'ff')
    put(f'{m}|{c}|ff|sd', _np.std(ff, ddof=1))

for key in ('full@compressed|delta|max', 'full@compressed|delta|min'):   # 'drops by x to y points': size of the drop
    put(key + '|abs', abs(K[key][0]), K[key][1])
for key in [k for k in K if k.endswith('|cut')] + [k for k in K if k.endswith('|delta')]:
    put(key + '|abs', abs(K[key][0]), K[key][1])


def vn(v, d):
    s = f'{abs(v):,.{d}f}'.replace(',', ' ').replace('.', ',').replace(' ', '.')
    return ('-' if v < 0 and float(s.replace('.', '').replace(',', '.')) != 0 else '') + s


out = ['% Generated by python -m experiments.paper_numbers; do not edit. Every number of the paper comes from here.',
       r'\makeatletter',
       r'\newcommand{\R}[1]{\ifcsname R@#1\endcsname\csname R@#1\endcsname\else\textcolor{red}{??#1??}\fi}',
       r'\newcommand{\Rs}[1]{\ifcsname Rs@#1\endcsname\csname Rs@#1\endcsname\else\textcolor{red}{??#1??}\fi}',
       r'\newcommand{\Rraw}[1]{\ifcsname Rr@#1\endcsname\csname Rr@#1\endcsname\else 0\fi}']
for key, (v, d) in sorted(K.items()):
    k = key.replace('#', '')
    out.append(rf'\expandafter\def\csname R@{k}\endcsname{{{vn(v, d)}}}')
    out.append(rf'\expandafter\def\csname Rs@{k}\endcsname{{{("+" if v > 0 else "") + vn(v, d)}}}')
    out.append(rf'\expandafter\def\csname Rr@{k}\endcsname{{{v:.{d}f}}}')
out.append(r'\makeatother')
(ROOT / 'outputs').mkdir(exist_ok=True)
(ROOT / 'outputs' / 'numbers.tex').write_text('\n'.join(out) + '\n', encoding='utf-8')
print(len(K), 'keys ->', ROOT / 'outputs' / 'numbers.tex')
