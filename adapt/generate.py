"""Generate the (G, K, D, A) dataset of the pick-and-place feasibility problem.

    python -m adapt.generate [--out data] [--seed 0] [--small]

How a row is made. Every quantity has one fixed distribution:
    limits theta_j    N(k0, (0.3 k0)^2) truncated to [0.6, 1.8] k0
    object o, mass    o uniform over the vocabulary; mass N(m_o, (0.3 m_o)^2) truncated to [0.6, 1.8] m_o
    reach_pick/place  uniform on [0.10, 1.20] m (base parked as close as the furniture allows)
    place B, height   B uniform over the places; height uniform on B's height range
    clearance         n readings, each uniform on [0.10, 2.50] m
    route, battery    length uniform on [3, 60] m; battery uniform on the integers 5..100
First a pattern is chosen: the answer y and a key rule k (uniform over the six rules), then for each rule a band of its
normalized slack s (near: 0.01 <= |s| < 0.10; far: |s| >= 0.10). The quantities of each rule are drawn from their fixed
distribution until s falls in the band (rejection sampling: the pattern only decides which draws are kept). Feasible
rows: rule k is near its limit with p 0.5; infeasible rows: k is violated (near with p 0.5), plus 0, 1 or 2 other rules
(p 0.60, 0.25, 0.15). Every other rule holds with any slack, the same band for both answers, so the rules that do not
decide carry no hint of the answer. A is then
computed by the rules from the final rounded numbers and must agree with the pattern; no |s_i| is below 0.01 (no ties).

Splits, per (answer, key rule) bucket (12 buckets):
    train 200, val 40, test 80     main recipe: half the key rules near their limit, catalog of 2-5 objects, 4-8 readings
    near_threshold       40        the key rule within 5% of its limit, and the only one deciding
    d_pairs              40 pairs per rule with a D quantity (5 rules): same G and K, one measurement moved across its
                                   limit
    k_pairs              35 pairs per rule (6 rules): same G and D, one limit of K moved across the measurement
    g_pairs             200 pairs: same K and D, the requested object swapped for another one of the catalog across the
                                   payload; the two objects have typical masses within a factor 1.4, so only the
                                   catalog decides
    unseen               40        held-out wordings, objects and pick-up places
    longer_than_train    20        catalog of 8 objects and 16 clearance readings (longer than any training row)
Every row passes verify.verify; no D is used twice across the whole set (pairs share theirs by design).
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .problem import (ALL_PLACES, D_RANGE, DEFAULT, DIGITS, FACTOR, HELD_OUT_OBJECTS, HELD_OUT_PLACES,
                      HELD_OUT_TEMPLATES, LIMITS, N_READINGS, OBJECTS, PLACES, RANGE, RULES, SIGMA, TEMPLATES,
                      ALL_OBJECTS, flip_margin, make_A, make_G, make_K, slack, violated)
from .verify import verify

NEAR, MIN_S = 0.10, 0.01
TRIES = 3000
P_EXTRA = [(0, 0.60), (1, 0.25), (2, 0.15)]
SIMILAR = 1.4               # g_pairs: typical masses of the two objects within this factor


# ---- fixed distributions ------------------------------------------------------------------------------------------------
def tnormal(rng, mean, digits):
    lo, hi = mean * FACTOR[0], mean * FACTOR[1]
    while True:
        v = round(float(rng.normal(mean, SIGMA * mean)), digits)
        if lo - 1e-9 <= v <= hi + 1e-9:
            return int(v) if digits == 0 else v


def sample_limits(rng):
    return {k: tnormal(rng, DEFAULT[k], DIGITS[k]) for k in LIMITS}


def sample_mass(rng, o):
    return tnormal(rng, ALL_OBJECTS[o], 2)


def u(rng, lo, hi, digits):
    return round(float(rng.uniform(lo, hi)), digits)


def draw(rule, rng, ctx):
    """One draw of the quantities of a rule from their fixed distribution; returns the fields it sets."""
    if rule == 'payload':
        o = ctx['objects'][int(rng.integers(len(ctx['objects'])))]
        return {'object': o, 'mass': sample_mass(rng, o)}
    if rule in ('reach_pick', 'reach_place'):
        return {rule: u(rng, *D_RANGE[rule], 2)}
    if rule == 'height':
        b = ctx['places_b'][int(rng.integers(len(ctx['places_b'])))]
        return {'to': b, 'place_height': u(rng, *ALL_PLACES[b], 2)}
    if rule == 'clearance':
        return {'clearance': [u(rng, *D_RANGE['clearance'], 2) for _ in range(ctx['n'])]}
    return {'route_length': u(rng, *D_RANGE['route_length'], 1), 'battery': int(rng.integers(5, 101))}


def assemble(f, limits, ctx, rng):
    """(G, K, D) from the drawn fields; the catalog holds the requested object plus distinct others."""
    o = f['object']
    others = [x for x in ctx['catalog_pool'] if x != o]
    rng.shuffle(others)
    names = [o] + others[:ctx['catalog'] - 1]
    rng.shuffle(names)
    masses = {x: (f['mass'] if x == o else sample_mass(rng, x)) for x in names}
    a_pool = [p for p in ctx['places_a'] if p != f['to']]
    G = make_G(o, a_pool[int(rng.integers(len(a_pool)))], f['to'], ctx['templates'][int(rng.integers(len(ctx['templates'])))])
    K = make_K(limits, masses)
    D = {k: f[k] for k in ('reach_pick', 'reach_place', 'place_height', 'clearance', 'route_length', 'battery')}
    return G, K, D


def s_of(rule, f, limits):
    """Normalized slack of one rule from the drawn fields alone (same formulas as problem.slack)."""
    t = limits
    if rule == 'payload':
        return (t['payload'] - f['mass']) / t['payload']
    if rule in ('reach_pick', 'reach_place'):
        return (t['reach'] - f[rule]) / t['reach']
    if rule == 'height':
        return (t['height'] - f['place_height']) / t['height']
    if rule == 'clearance':
        return (min(f['clearance']) - t['clearance']) / t['clearance']
    nd = t['energy'] * f['route_length'] + t['reserve']
    return (f['battery'] - nd) / nd


BANDS = {'ok_near': (MIN_S, NEAR), 'ok': (MIN_S, math.inf), 'bad_near': (-NEAR, -MIN_S), 'bad': (-math.inf, -MIN_S)}


def in_band(s, band):
    lo, hi = BANDS[band]
    return lo <= s < hi if band.startswith('ok') else lo < s <= hi


def draw_band(rule, band, rng, ctx, limits):
    for _ in range(TRIES):
        f = draw(rule, rng, ctx)
        if in_band(s_of(rule, f, limits), band):
            return f
    return None


# ---- one row ------------------------------------------------------------------------------------------------------------
def pattern(rng, y, k, near):
    """Band of every rule. near: True / False / None (None: near with p 0.5)."""
    is_near = bool(rng.random() < 0.5) if near is None else near
    bands = {k: ('ok_near' if is_near else 'ok') if y else ('bad_near' if is_near else 'bad')}
    if y:
        bands.update({r: 'ok' for r in RULES if r != k})
    else:
        n_extra = 0 if near else int(rng.choice([n for n, _ in P_EXTRA], p=[p for _, p in P_EXTRA]))
        extra = set(rng.choice([r for r in RULES if r != k], size=n_extra, replace=False)) if n_extra else set()
        bands.update({r: ('bad' if r in extra else 'ok') for r in RULES if r != k})
    return bands


def sample_fields(rng, k, bands, ctx, limits):
    f = draw_band(k, bands[k], rng, ctx, limits)
    if f is None:
        return None
    for r in RULES:
        if r == k:
            continue
        fr = draw_band(r, bands[r], rng, ctx, limits)
        if fr is None:
            return None
        f.update(fr)
    return f


def finish(G, K, D, split, rid, ctx, **extra):
    A = make_A(G, K, D)
    s = slack(G, K, D)
    fm = flip_margin(G, K, D)
    row = {'id': rid, 'split': split, 'G': G, 'K': K, 'D': D, 'A': A,
           'key_rule': extra.pop('key_rule', None), 'slack': {r: round(v, 4) for r, v in s.items()},
           'flip_margin': None if math.isinf(fm) else round(fm, 4), 'near': bool(fm < NEAR),
           'n_violated': len(A['violated']), 'catalog_size': len(K['masses']), 'n_readings': len(D['clearance']),
           'k_dependent': k_dependent(G, K, D)}
    row.update(extra)
    return row


def make_row(rng, y, k, ctx, split, rid, near=None):
    limits = sample_limits(rng)
    bands = pattern(rng, y, k, near)
    f = sample_fields(rng, k, bands, ctx, limits)
    if f is None:
        return None, 'no draw in the band'
    G, K, D = assemble(f, limits, ctx, rng)
    row = finish(G, K, D, split, rid, ctx, key_rule=k)
    if row['A']['feasible'] != y or (not y and k not in row['A']['violated']):
        return None, 'answer differs from the pattern'
    if min(abs(v) for v in slack(G, K, D).values()) < MIN_S:
        return None, 'a slack below 0.01'
    return row, None


def k_dependent(G, K, D, points=9):
    """True if moving one limit of K, or the mass of the requested object, inside its range flips the answer."""
    base = not violated(G, K, D)
    for key in LIMITS + ['mass']:
        lo, hi = MASS_RANGE_OF(G['object']) if key == 'mass' else RANGE[key]
        for v in np.linspace(lo, hi, points):
            if key == 'mass':
                K2 = dict(K, masses=dict(K['masses'], **{G['object']: float(v)}))
            else:
                K2 = dict(K, limits=dict(K['limits'], **{key: float(v)}))
            if (not violated(G, K2, D)) != base:
                return True
    return False


def MASS_RANGE_OF(o):
    m = ALL_OBJECTS[o]
    return m * FACTOR[0], m * FACTOR[1]


# ---- contexts -----------------------------------------------------------------------------------------------------------
def context(rng, kind='main'):
    train_objects = list(OBJECTS)
    ctx = {'objects': train_objects, 'catalog_pool': train_objects, 'places_a': list(PLACES), 'places_b': list(PLACES),
           'templates': TEMPLATES, 'n': int(rng.integers(N_READINGS[0], N_READINGS[1] + 1)),
           'catalog': int(rng.integers(2, 6))}
    if kind == 'heldout':
        ctx.update(objects=list(HELD_OUT_OBJECTS), catalog_pool=list(ALL_OBJECTS), places_a=list(HELD_OUT_PLACES),
                   templates=HELD_OUT_TEMPLATES)
    elif kind == 'longer':
        ctx.update(n=16, catalog=8)
    return ctx


class Registry:
    """Every D appears once in the whole set (a pair counts as one claim)."""

    def __init__(self):
        self.seen = set()

    @staticmethod
    def key(D):
        return (D['reach_pick'], D['reach_place'], D['place_height'], tuple(D['clearance']), D['route_length'], D['battery'])

    def claim(self, *Ds):
        keys = {self.key(D) for D in Ds}
        if keys & self.seen:
            return False
        self.seen |= keys
        return True


def build_buckets(rng, reg, rej, split, per, kind='main', near=None, near_max=None):
    """per rows for every (answer, key rule); near_max: the key rule within this slack and the only deciding rule."""
    rows = []
    for y in (True, False):
        for k in RULES:
            got = 0
            while got < per:
                row, why = make_row(rng, y, k, context(rng, kind), split, f'{split}/{len(rows)}', near=near)
                if row is None:
                    rej[why] += 1; continue
                if near_max is not None and (row['flip_margin'] is None or row['flip_margin'] >= near_max):
                    rej['not near enough'] += 1; continue
                r = verify(row)
                if r:
                    rej[f'verify: {r}'] += 1; continue
                if not reg.claim(row['D']):
                    rej['D already used'] += 1; continue
                rows.append(row); got += 1
    return rows


# ---- pairs ---------------------------------------------------------------------------------------------------------------
D_RULES = ['reach_pick', 'reach_place', 'height', 'clearance', 'battery']


def build_dpairs(rng, reg, rej, per):
    """Same G and K; only the measurement of rule k moves, from just inside its limit to just outside."""
    rows, pid = [], 0
    for k in D_RULES:
        got = 0
        while got < per:
            ctx = context(rng)
            limits = sample_limits(rng)
            bands = {r: 'ok' for r in RULES}
            bands[k] = 'ok_near'
            f = sample_fields(rng, k, bands, ctx, limits)
            if f is None:
                rej['no draw in the band'] += 1; continue
            f2 = dict(f)
            if k == 'clearance':
                i = int(np.argmin(f['clearance']))
                for _ in range(TRIES):
                    v = u(rng, limits['clearance'] * (1 - NEAR), limits['clearance'] * (1 - MIN_S), 2)
                    f2['clearance'] = f['clearance'][:i] + [v] + f['clearance'][i + 1:]
                    if in_band(s_of(k, f2, limits), 'bad_near'):
                        break
            else:
                ctx_b = dict(ctx, places_b=[f['to']])                  # height: the same place B
                for _ in range(TRIES):
                    d = draw(k, rng, ctx_b)
                    if k == 'battery':
                        d = {'battery': d['battery'], 'route_length': f['route_length']}
                    f2.update(d)
                    if in_band(s_of(k, f2, limits), 'bad_near'):
                        break
            if not in_band(s_of(k, f2, limits), 'bad_near'):
                rej['pair: no near draw on the other side'] += 1; continue
            G, K, D = assemble(f, limits, ctx, rng)
            D2 = dict(D, **{q: f2[q] for q in D})
            pair = [finish(G, K, D, 'd_pairs', f'd_pairs/{pid}/0', ctx, key_rule=k, pair_id=pid, changed=k),
                    finish(G, K, D2, 'd_pairs', f'd_pairs/{pid}/1', ctx, key_rule=k, pair_id=pid, changed=k)]
            if [r['A']['feasible'] for r in pair] != [True, False] or pair[1]['A']['violated'] != [k]:
                rej['pair: answer did not flip on rule k alone'] += 1; continue
            bad = [verify(r) for r in pair]
            if any(bad):
                rej[f'verify: {next(b for b in bad if b)}'] += 1; continue
            if not reg.claim(D, D2):
                rej['D already used'] += 1; continue
            rows += pair; got += 1; pid += 1
    return rows


def build_kpairs(rng, reg, rej, per):
    """Same G and D; one limit of K (for the battery: the reserve or the energy per meter) moves so rule k fails."""
    rows, pid = [], 0
    for k in RULES:
        got = 0
        while got < per:
            ctx = context(rng)
            row, why = make_row(rng, True, k, ctx, 'k_pairs', f'k_pairs/{pid}/0')
            if row is None:
                rej[why] += 1; continue
            G, K, D = row['G'], row['K'], row['D']
            key = LIMIT_KEY[k] if k != 'battery' else ('reserve', 'energy')[int(rng.integers(2))]
            lo, hi = RANGE[key]
            K2 = None
            for _ in range(TRIES):
                v = u(rng, lo, hi, DIGITS[key])
                v = int(v) if DIGITS[key] == 0 else v
                cand = dict(K, limits=dict(K['limits'], **{key: v}))
                cand = make_K(cand['limits'], K['masses'])
                s = slack(G, cand, D)
                if s[k] <= -MIN_S and min(abs(x) for x in s.values()) >= MIN_S:
                    K2 = cand; break
            if K2 is None:
                rej['pair: the limit cannot fail inside its range'] += 1; continue
            pair = [finish(G, K, D, 'k_pairs', f'k_pairs/{pid}/0', ctx, key_rule=k, pair_id=pid, changed=key),
                    finish(G, K2, D, 'k_pairs', f'k_pairs/{pid}/1', ctx, key_rule=k, pair_id=pid, changed=key)]
            if [r['A']['feasible'] for r in pair] != [True, False] or k not in pair[1]['A']['violated']:
                rej['pair: answer did not flip'] += 1; continue
            bad = [verify(r) for r in pair]
            if any(bad):
                rej[f'verify: {next(b for b in bad if b)}'] += 1; continue
            if not reg.claim(D):
                rej['D already used'] += 1; continue
            rows += pair; got += 1; pid += 1
    return rows


LIMIT_KEY = {'payload': 'payload', 'reach_pick': 'reach', 'reach_place': 'reach', 'height': 'height',
             'clearance': 'clearance'}


def build_gpairs(rng, reg, rej, n):
    """Same K and D; G asks for a light object of the catalog (payload holds) or a heavy one (payload fails)."""
    rows, pid = [], 0
    while pid < n:
        ctx = context(rng)
        ctx['catalog'] = max(ctx['catalog'], 2)
        limits = sample_limits(rng)
        bands = {r: 'ok' for r in RULES}
        f = sample_fields(rng, 'payload', bands, ctx, limits)
        if f is None:
            rej['no draw in the band'] += 1; continue
        # the other object weighs about the same as a rule (typical masses within a factor SIMILAR), so the name of the
        # object does not tell which one is too heavy: only the catalog of K does
        m1 = ALL_OBJECTS[f['object']]
        alike = [x for x in ctx['objects'] if x != f['object'] and max(m1, ALL_OBJECTS[x]) / min(m1, ALL_OBJECTS[x]) <= SIMILAR]
        heavy = draw_band('payload', 'bad', rng, dict(ctx, objects=alike), limits) if alike else None
        if heavy is None:
            rej['no object of similar mass over the payload'] += 1; continue
        o1, o2 = f['object'], heavy['object']
        G, _, D = assemble(f, limits, ctx, rng)
        others = [x for x in ctx['catalog_pool'] if x not in (o1, o2)]
        rng.shuffle(others)
        names = [o1, o2] + others[:ctx['catalog'] - 2]
        rng.shuffle(names)
        K = make_K(limits, {x: f['mass'] if x == o1 else heavy['mass'] if x == o2 else sample_mass(rng, x) for x in names})
        template = next(t for t in ctx['templates'] if t.format(o=o1, a=G['from'], b=G['to']) == G['text'])
        G2 = make_G(o2, G['from'], G['to'], template)
        pair = [finish(G, K, D, 'g_pairs', f'g_pairs/{pid}/0', ctx, key_rule='payload', pair_id=pid, changed='object'),
                finish(G2, K, D, 'g_pairs', f'g_pairs/{pid}/1', ctx, key_rule='payload', pair_id=pid, changed='object')]
        if [r['A']['feasible'] for r in pair] != [True, False] or pair[1]['A']['violated'] != ['payload']:
            rej['pair: answer did not flip on the payload alone'] += 1; continue
        bad = [verify(r) for r in pair]
        if any(bad):
            rej[f'verify: {next(b for b in bad if b)}'] += 1; continue
        if not reg.claim(D):
            rej['D already used'] += 1; continue
        rows += pair; pid += 1
    return rows


# ---- driver ------------------------------------------------------------------------------------------------------------
def write(path, rows):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m adapt.generate')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent.parent / 'data'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--small', action='store_true', help='tiny sizes for a smoke test')
    a = ap.parse_args(argv)
    sizes = dict(train=200, val=40, test=80, near_threshold=40, d_pairs=40, k_pairs=35, g_pairs=200, unseen=40,
                 longer_than_train=20)
    if a.small:
        sizes = dict(train=4, val=2, test=2, near_threshold=2, d_pairs=2, k_pairs=2, g_pairs=4, unseen=2,
                     longer_than_train=1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = Registry()
    man = {'seed': a.seed, 'sizes_per_bucket_or_pairs': sizes, 'limit_ranges': RANGE, 'files': {}, 'rejected': {}}
    stages = [('train', 1, build_buckets, ('train', sizes['train']), {}),
              ('val', 2, build_buckets, ('val', sizes['val']), {}),
              ('test', 3, build_buckets, ('test', sizes['test']), {}),
              ('near_threshold', 4, build_buckets, ('near_threshold', sizes['near_threshold']),
               dict(near=True, near_max=0.05)),
              ('d_pairs', 5, build_dpairs, (sizes['d_pairs'],), {}),
              ('k_pairs', 6, build_kpairs, (sizes['k_pairs'],), {}),
              ('g_pairs', 7, build_gpairs, (sizes['g_pairs'],), {}),
              ('unseen', 8, build_buckets, ('unseen', sizes['unseen']), dict(kind='heldout')),
              ('longer_than_train', 9, build_buckets, ('longer_than_train', sizes['longer_than_train']),
               dict(kind='longer'))]
    for name, offset, fn, args, kw in stages:
        rng = np.random.default_rng([a.seed, offset])           # one stream per split
        rej = Counter()
        rows = fn(rng, reg, rej, *args, **kw)
        man['files'][f'{name}.jsonl'] = write(out / f'{name}.jsonl', rows)
        man['rejected'][name] = dict(rej.most_common())
        print(f'[{name}] {len(rows)} rows; draws rejected {sum(rej.values())}', flush=True)
    man['total_rows'] = sum(man['files'].values())
    (out / 'manifest.json').write_text(json.dumps(man, indent=2), encoding='utf-8')
    print('total', man['total_rows'])
    return man


def extra_train(argv=None):
    """A larger training file next to the existing data, which stays as it is:

        python -c "from adapt.generate import extra_train; extra_train()"   -> data/train_x4.jsonl (800 rows per bucket)

    Same recipe as train.jsonl (same distributions, same 12 buckets), its own random stream, and no D that any existing
    file uses (the registry is filled from every file in the folder first). Used to ask whether more training data
    teaches the minimum and the arithmetic."""
    ap = argparse.ArgumentParser(prog='extra_train')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent.parent / 'data'))
    ap.add_argument('--name', default='train_x4')
    ap.add_argument('--per', type=int, default=800)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--stream', type=int, default=11, help='random stream; 1-9 are the main files')
    a = ap.parse_args(argv)
    out = Path(a.out)
    reg = Registry()
    for f in sorted(out.glob('*.jsonl')):
        if f.stem == a.name:
            continue
        for line in open(f, encoding='utf-8'):
            reg.seen.add(Registry.key(json.loads(line)['D']))
    n_before = len(reg.seen)
    rng = np.random.default_rng([a.seed, a.stream])
    rej = Counter()
    rows = build_buckets(rng, reg, rej, a.name, a.per)
    n = write(out / f'{a.name}.jsonl', rows)
    man = {'seed': a.seed, 'stream': a.stream, 'per_bucket': a.per, 'rows': n, 'd_already_used': n_before,
           'rejected': dict(rej.most_common()), 'recipe': 'build_buckets, as train.jsonl'}
    (out / f'manifest_{a.name}.json').write_text(json.dumps(man, indent=2), encoding='utf-8')
    print(f'[{a.name}] {n} rows; draws rejected {sum(rej.values())}; {n_before} D of other files excluded')
    return man


if __name__ == '__main__':
    main()
