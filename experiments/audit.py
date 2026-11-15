"""Independent audit of data/: the answer of every row is recomputed from the TEXT the model reads (the rule sentences
and the catalog sentence of K, the numbers of D), with its own parser and its own arithmetic, without importing the
generator. Then structure, splits, pairs, uniqueness and held-out separation are checked.

    python -m experiments.audit [--data data]
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--data', default=str(Path(__file__).resolve().parent.parent / 'data'))
a = ap.parse_args()
DATA = Path(a.data)
FILES = ['train', 'val', 'test', 'near_threshold', 'd_pairs', 'k_pairs', 'g_pairs', 'unseen', 'longer_than_train']
rows = {f: [json.loads(l) for l in open(DATA / f'{f}.jsonl', encoding='utf-8')] for f in FILES}
checks = []


def check(name, ok, detail=''):
    checks.append((name, bool(ok), detail))


NUM = r'(\d+(?:\.\d+)?)'


def blank(G):
    """The wording with the object name (whole words, first mention) replaced by '#'."""
    return re.sub(rf'\b{re.escape(G["object"])}\b', '#', G['text'], count=1)


def limits_from_text(rules):
    t = {}
    t['payload'] = float(re.fullmatch(rf'The robot can lift objects of at most {NUM} kg\.', rules['payload']).group(1))
    r1 = float(re.fullmatch(rf'The arm can pick up an object at most {NUM} m from its base\.', rules['reach_pick']).group(1))
    r2 = float(re.fullmatch(rf'The arm can put an object down at most {NUM} m from its base\.', rules['reach_place']).group(1))
    t['reach_pick'], t['reach_place'] = r1, r2
    t['height'] = float(re.fullmatch(rf'The arm can put an object on a surface at most {NUM} m high\.', rules['height']).group(1))
    t['clearance'] = float(re.fullmatch(rf'The robot needs at least {NUM} m of clearance at every point of its route\.',
                                        rules['clearance']).group(1))
    m = re.fullmatch(rf'Driving uses {NUM}% of the battery per meter, and at least {NUM}% must remain at the end of the '
                     r'route\.', rules['battery'])
    t['energy'], t['reserve'] = float(m.group(1)), float(m.group(2))
    return t


def masses_from_text(text):
    body = re.fullmatch(r'Known object masses: (.*)\.', text).group(1)
    out = {}
    for part in body.split('; '):
        m = re.fullmatch(rf'the (.+) weighs {NUM} kg', part)
        out[m.group(1)] = float(m.group(2))
    return out


def answer(row):
    """(feasible, violated rules) from the text of K and the numbers of D."""
    t, ms, D = limits_from_text(row['K']['rules']), masses_from_text(row['K']['masses_text']), row['D']
    named = row['G']['object']
    bad = []
    # use integer arithmetic in hundredths to avoid float ties (every number has at most 2 decimals; energy*length 3)
    c = lambda x: round(x * 1000)
    if c(ms[named]) > c(t['payload']):
        bad.append('payload')
    if c(D['reach_pick']) > c(t['reach_pick']):
        bad.append('reach_pick')
    if c(D['reach_place']) > c(t['reach_place']):
        bad.append('reach_place')
    if c(D['place_height']) > c(t['height']):
        bad.append('height')
    if c(min(D['clearance'])) < c(t['clearance']):
        bad.append('clearance')
    if round(D['battery'] * 1000) < round(t['energy'] * 1000 * D['route_length']) + c(t['reserve']):
        bad.append('battery')
    return not bad, bad


# ---- 1. answers recomputed from the text --------------------------------------------------------------------------------
n_rows = sum(len(v) for v in rows.values())
mism = [(f, r['id']) for f in FILES for r in rows[f] if answer(r) != (r['A']['feasible'], r['A']['violated'])]
check('answer = rules applied to the text of K and the numbers of D (own parser, integer arithmetic)', not mism,
      f'{n_rows - len(mism)}/{n_rows}' + (f'; first {mism[:3]}' if mism else ''))
check('action and plan follow the answer', all((r['A']['action'] == 'Execute') == r['A']['feasible'] and
                                                ((r['A']['plan'] is None) != r['A']['feasible']) for f in FILES for r in rows[f]))
check('the requested object is named in the wording and listed in the catalog',
      all(r['G']['object'] in r['G']['text'] and r['G']['object'] in masses_from_text(r['K']['masses_text'])
          for f in FILES for r in rows[f]))
check('the two reach sentences carry the same limit',
      all(limits_from_text(r['K']['rules'])['reach_pick'] == limits_from_text(r['K']['rules'])['reach_place']
          for f in FILES for r in rows[f]))
check('every D has exactly the six measurements', all(list(r['D']) == ['reach_pick', 'reach_place', 'place_height',
                                                                       'clearance', 'route_length', 'battery']
                                                      for f in FILES for r in rows[f]))
ties = [r['id'] for f in FILES for r in rows[f] if min(abs(v) for v in r['slack'].values()) < 0.0099]
check('no measurement within 1% of its limit (no ties)', not ties, f'{len(ties)} rows')

# ---- 1b. the most compressed prompt still holds the answer ----------------------------------------------------------------
from adapt.prompts import user_text  # noqa: E402  (the text the model reads; the parser below is this file's own)


def answer_from_compressed(text):
    """Decode the compressed prompt alone (names and numbers, no rule sentences) and apply the six rules."""
    lines = text.split('\n')
    request = lines[0][len('Request: '):]
    kv = dict(l.split(': ', 1) for l in lines[1:])
    D = {k: json.loads(kv[k]) for k in ('reach_pick', 'reach_place', 'place_height', 'clearance', 'route_length', 'battery')}
    t = {k: float(v) for k, v in (p.rsplit(' ', 1) for p in kv['limits'].split(', '))}
    ms = {k: float(v) for k, v in (p.rsplit(' ', 1) for p in kv['masses'].split(', '))}
    named = [o for o in ms if re.search(rf'\b{re.escape(o)}\b', request)]
    o = max(named, key=len)                                   # 'stack of plates' over 'plate'
    c = lambda x: round(x * 1000)
    bad = [r for r, fails in (
        ('payload', c(ms[o]) > c(t['payload'])), ('reach_pick', c(D['reach_pick']) > c(t['reach'])),
        ('reach_place', c(D['reach_place']) > c(t['reach'])), ('height', c(D['place_height']) > c(t['height'])),
        ('clearance', c(min(D['clearance'])) < c(t['clearance'])),
        ('battery', round(D['battery'] * 1000) < round(t['energy'] * 1000 * D['route_length']) + c(t['reserve']))) if fails]
    return not bad, bad


mism3 = [r['id'] for f in FILES for r in rows[f]
         if answer_from_compressed(user_text(r, 'compressed')) != (r['A']['feasible'], r['A']['violated'])]
check('the compressed prompt alone (names and numbers) determines the answer: decoded and recomputed', not mism3,
      f'{n_rows - len(mism3)}/{n_rows}' + (f'; first {mism3[:3]}' if mism3 else ''))

# ---- 2. balance --------------------------------------------------------------------------------------------------------
for f in ('train', 'val', 'test', 'near_threshold', 'unseen', 'longer_than_train'):
    c = Counter((r['A']['feasible'], r['key_rule']) for r in rows[f])
    check(f'{f}: every (answer, key rule) bucket has the same size', len(c) == 12 and len(set(c.values())) == 1,
          f'{len(c)} buckets of {set(c.values())}')
for f in ('d_pairs', 'k_pairs', 'g_pairs'):
    by = {}
    for r in rows[f]:
        by.setdefault(r['pair_id'], []).append(r)
    ok = all(len(p) == 2 and sorted(x['A']['feasible'] for x in p) == [False, True] for p in by.values())
    check(f'{f}: every pair has one feasible and one infeasible row', ok, f'{len(by)} pairs')
    same = {'d_pairs': ('G', 'K'), 'k_pairs': ('G', 'D'), 'g_pairs': ('K', 'D')}[f]
    diff = {'d_pairs': 'D', 'k_pairs': 'K', 'g_pairs': 'G'}[f]
    check(f'{f}: the two rows share {" and ".join(same)} and differ in {diff} only',
          all(all(p[0][x] == p[1][x] for x in same) and p[0][diff] != p[1][diff] for p in by.values()))
    if f == 'd_pairs':
        def n_changed(p):
            a_, b_ = p[0]['D'], p[1]['D']
            return sum(a_[k] != b_[k] if k != 'clearance' else sum(x != y for x, y in zip(a_[k], b_[k])) for k in a_)
        check('d_pairs: exactly one number of D differs', all(n_changed(p) == 1 for p in by.values()))
    if f == 'k_pairs':
        check('k_pairs: exactly one limit of K differs',
              all(sum(p[0]['K']['limits'][k] != p[1]['K']['limits'][k] for k in p[0]['K']['limits']) == 1 for p in by.values()))
    if f == 'g_pairs':
        check('g_pairs: both objects are in the shared catalog and the wordings differ only in the object',
              all(p[0]['G']['object'] in p[0]['K']['masses'] and p[1]['G']['object'] in p[0]['K']['masses'] and
                  blank(p[0]['G']) == blank(p[1]['G']) for p in by.values()))

# ---- 3. separation ------------------------------------------------------------------------------------------------------
dkey = lambda D: json.dumps(D, sort_keys=True)
owner = {}
clash = 0
for f in FILES:
    seen_here = set()
    for r in rows[f]:
        k = dkey(r['D'])
        if k in seen_here and f not in ('k_pairs', 'g_pairs'):
            clash += 1
        if k in owner and owner[k] != f:
            clash += 1
        owner.setdefault(k, f); seen_here.add(k)
check('no D appears in two files, nor twice in one file (except the shared D of a K- or G-pair)', clash == 0, f'{clash}')
train_text = {r['G']['text'] for r in rows['train']}
held = rows['unseen']
check('unseen: no held-out wording, object or pick-up place appears in train/val/test',
      not ({r['G']['object'] for r in held} & {r['G']['object'] for f in ('train', 'val', 'test') for r in rows[f]})
      and not ({r['G']['from'] for r in held} & {r['G']['from'] for f in ('train', 'val', 'test') for r in rows[f]})
      and not ({r['G']['text'] for r in held} & train_text))
check('longer_than_train: catalog of 8 objects and 16 readings in every row; at most 5 and 8 in train',
      all(len(r['K']['masses']) == 8 and len(r['D']['clearance']) == 16 for r in rows['longer_than_train'])
      and max(len(r['K']['masses']) for r in rows['train']) <= 5 and max(len(r['D']['clearance']) for r in rows['train']) <= 8)
check('near_threshold: every row is decided by one rule within 5% of its limit',
      all(r['flip_margin'] is not None and r['flip_margin'] < 0.05 for r in rows['near_threshold']))

bad = [c for c in checks if not c[1]]
L = ['# Independent audit of the dataset', '', '> Generated by `python -m experiments.audit`. The answer is recomputed from the **wording** of K '
     '(the rule sentences and the mass sentence) and the numbers of D, with its own parser and arithmetic (integer thousandths), without the '
     'code of the generator.', '', f'**{len(checks) - len(bad)}/{len(checks)} checks passed.**', '', '| Check | Result | Details |',
     '|---|---|---|']
L += [f"| {n} | {'pass' if ok else '**FAIL**'} | {d} |" for n, ok, d in checks]
(DATA / 'AUDIT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
for n, ok, d in checks:
    print('ok  ' if ok else 'FAIL', n, d)
sys.exit(1 if bad else 0)
