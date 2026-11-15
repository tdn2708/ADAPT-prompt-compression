"""Tests of the generator, the validity check and the prompt levels (run: python -m pytest -q)."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from adapt import generate
from adapt.problem import RULES, feasible, flip_margin, make_A, make_G, make_K, violated
from adapt.prompts import LEVELS, user_text
from adapt.verify import verify

ROOT = Path(__file__).resolve().parent.parent
FILES = ['train', 'val', 'test', 'near_threshold', 'd_pairs', 'k_pairs', 'g_pairs', 'unseen', 'longer_than_train']


@pytest.fixture(scope='module')
def small(tmp_path_factory):
    out = tmp_path_factory.mktemp('data')
    generate.main(['--small', '--out', str(out)])
    return out, {f: [json.loads(l) for l in open(out / f'{f}.jsonl', encoding='utf-8')] for f in FILES}


def test_every_row_valid(small):
    _, rows = small
    assert all(verify(r) is None for f in FILES for r in rows[f])


def test_independent_audit_passes(small):
    out, _ = small
    r = subprocess.run([sys.executable, '-m', 'experiments.audit', '--data', str(out)], cwd=ROOT, capture_output=True,
                       text=True, encoding='utf-8')
    assert r.returncode == 0, r.stdout + r.stderr


def test_deterministic(tmp_path, small):
    generate.main(['--small', '--out', str(tmp_path)])
    out, _ = small
    for f in FILES:
        assert (tmp_path / f'{f}.jsonl').read_bytes() == (out / f'{f}.jsonl').read_bytes()


def base_row():
    K = make_K({'payload': 1.5, 'reach': 0.8, 'height': 1.2, 'clearance': 0.35, 'energy': 0.5, 'reserve': 20},
               {'kettle': 1.3, 'cup': 0.3})
    G = make_G('kettle', 'dining table', 'wall shelf', 'Move the {o} from the {a} to the {b}.')
    D = {'reach_pick': 0.6, 'reach_place': 0.5, 'place_height': 1.1, 'clearance': [1.0, 0.5, 2.0, 0.9], 'route_length': 20.0,
         'battery': 60}
    return {'id': 'x', 'G': G, 'K': K, 'D': D, 'A': make_A(G, K, D)}


def test_rules_by_hand():
    r = base_row()
    assert feasible(r['G'], r['K'], r['D'])                      # 60 - (0.5 * 20 + 20) = 30 >= 0
    for field, value, rule in [('reach_pick', 0.81, 'reach_pick'), ('reach_place', 0.9, 'reach_place'),
                               ('place_height', 1.25, 'height'), ('battery', 29, 'battery')]:
        D = dict(r['D'], **{field: value})
        assert violated(r['G'], r['K'], D) == [rule]
    D = dict(r['D'], clearance=[1.0, 0.34, 2.0, 0.9])
    assert violated(r['G'], r['K'], D) == ['clearance']
    G = make_G('kettle', 'dining table', 'wall shelf', 'Move the {o} from the {a} to the {b}.')
    K = make_K(r['K']['limits'], {'kettle': 1.6, 'cup': 0.3})
    assert violated(G, K, r['D']) == ['payload']
    assert flip_margin(r['G'], r['K'], r['D']) == pytest.approx(min((1.5 - 1.3) / 1.5, (0.8 - 0.6) / 0.8, (0.8 - 0.5) / 0.8,
                                                                   (1.2 - 1.1) / 1.2, (0.5 - 0.35) / 0.35, 30 / 30))


@pytest.mark.parametrize('fault, part', [
    (lambda r: r['A'].update(feasible=not r['A']['feasible']), 'A'),
    (lambda r: r['K']['limits'].update(payload=1.2), 'K'),                     # limit changed, sentence not
    (lambda r: r['K']['masses'].pop('kettle'), 'K'),                           # requested object missing from the catalog
    (lambda r: r['D'].update(place_height=0.5), 'D'),                          # below the wall shelf's height range
    (lambda r: r['D'].update(battery=60.0), 'D'),                              # not an integer
    (lambda r: r['D'].update(clearance=[]), 'D'),
    (lambda r: r['G'].update(text='Move it.'), 'G'),
    (lambda r: r['G'].update(to='dining table'), 'G'),                         # same place twice
])
def test_fault_injection(fault, part):
    r = base_row()
    assert verify(r) is None
    fault(r)
    res = verify(r)
    assert res is not None and res.startswith(part), res


def test_levels_keep_every_number(small):
    _, rows = small
    import re
    for r in rows['test']:
        nums = {lv: sorted(re.findall(r'\d+(?:\.\d+)?', user_text(r, lv))) for lv in LEVELS}
        for lv in LEVELS[1:]:
            assert set(nums['compressed']) <= set(nums['full'])
        D_nums = [str(v) for k, v in r['D'].items() if k != 'clearance']
        assert all(n.rstrip('0').rstrip('.') in user_text(r, 'compressed') or n in user_text(r, 'compressed') for n in D_nums)
    lens = [len(user_text(rows['test'][0], lv)) for lv in LEVELS]
    assert lens == sorted(lens, reverse=True)


def test_pairs_flip(small):
    _, rows = small
    for f in ('d_pairs', 'k_pairs', 'g_pairs'):
        by = {}
        for r in rows[f]:
            by.setdefault(r['pair_id'], []).append(r)
        assert all(sorted(x['A']['feasible'] for x in p) == [False, True] for p in by.values())


def test_buckets_balanced(small):
    _, rows = small
    from collections import Counter
    c = Counter((r['A']['feasible'], r['key_rule']) for r in rows['train'])
    assert len(c) == 2 * len(RULES) and len(set(c.values())) == 1


def test_verify_does_not_mutate():
    r = base_row()
    r2 = copy.deepcopy(r)
    verify(r)
    assert r == r2


def test_control_levels(small):
    """compressed_mass_sentences = compressed with the catalog sentence; full_code = full plus the smallest clearance and
    the battery the trip needs. The two
    derived numbers must be right and must give the same verdicts as the rules (rounding to 2 decimals moves none)."""
    import re
    from adapt.problem import need, violated
    _, rows = small
    for f in FILES:
        for r in rows[f]:
            l3, l3m = user_text(r, 'compressed').split('\n'), user_text(r, 'compressed_mass_sentences').split('\n')
            assert l3[:-1] == l3m[:-1] and l3m[-1] == r['K']['masses_text']
            l0, l0d = user_text(r, 'full'), user_text(r, 'full_code')
            extra = [x for x in l0d.split('\n') if x not in l0.split('\n')]
            assert len(extra) == 2 and l0d.replace('\n'.join(extra) + '\n', '') == l0
            smallest = float(re.search(r'route, m: ([\d.]+)$', extra[0]).group(1))
            needed = float(re.search(r'%: ([\d.]+)$', extra[1]).group(1))
            assert smallest == min(r['D']['clearance'])
            assert abs(needed - need(r['K'], r['D'])) < 0.006
            bad = violated(r['G'], r['K'], r['D'])
            assert (smallest < r['K']['limits']['clearance']) == ('clearance' in bad)
            assert (r['D']['battery'] < needed) == ('battery' in bad)


def test_payload_sentence_levels(small):
    """compressed_payload_rule and compressed_payload_rule_mass_sentences add to compressed exactly the payload rule
    sentence (and, for the second, the catalog sentence), nothing else."""
    _, rows = small
    for f in FILES:
        for r in rows[f]:
            l3, l3r, l3mr = (user_text(r, lv).split('\n') for lv in ('compressed', 'compressed_payload_rule',
                                                                       'compressed_payload_rule_mass_sentences'))
            rule = r['K']['rules']['payload']
            assert l3r == l3[:-1] + [rule, l3[-1]]
            assert l3mr == l3[:-1] + [rule, r['K']['masses_text']]


def test_extra_train_shares_no_D(tmp_path):
    """extra_train writes a larger training file with the main recipe, sharing no D with any other file, and leaves the
    other files byte-identical."""
    from collections import Counter
    generate.main(['--small', '--out', str(tmp_path)])
    before = {f.name: f.read_bytes() for f in tmp_path.glob('*')}
    generate.extra_train(['--out', str(tmp_path), '--name', 'train_xs', '--per', '3'])
    assert all((tmp_path / n).read_bytes() == b for n, b in before.items())
    extra = [json.loads(l) for l in open(tmp_path / 'train_xs.jsonl', encoding='utf-8')]
    assert all(verify(r) is None for r in extra)
    assert set(Counter((r['A']['feasible'], r['key_rule']) for r in extra).values()) == {3}
    key = lambda r: json.dumps(r['D'], sort_keys=True)
    others = {key(json.loads(l)) for f in FILES for l in open(tmp_path / f'{f}.jsonl', encoding='utf-8')}
    assert not {key(r) for r in extra} & others


def test_cot_rationale_follows_the_rules():
    """The step-by-step answers of H8 are computed from the row: every line agrees with the rules on every row."""
    from adapt.cot import check_rationale, parse_answer, rationale
    for f in FILES:
        for l in open(ROOT / 'data' / f'{f}.jsonl', encoding='utf-8'):
            assert check_rationale(json.loads(l))
    assert parse_answer('x\nFeasible: True\nFeasible: False') is False and parse_answer('no answer') is None


def test_round4_level_and_paraphrases():
    """compressed_code adds the two computed lines to compressed and nothing else; every free-form request names the object and both places of its
    test row, and keeps that row's K, D and answer."""
    from adapt.problem import fmt, need
    test = {r['id']: r for r in (json.loads(l) for l in open(ROOT / 'data' / 'test.jsonl', encoding='utf-8'))}
    for r in list(test.values())[:50]:
        l3, l3d = user_text(r, 'compressed').split('\n'), user_text(r, 'compressed_code').split('\n')
        extra = [x for x in l3d if x not in l3]
        assert extra == [f"clearance_min: {fmt(min(r['D']['clearance']))}", f"battery_needed: {fmt(round(need(r['K'], r['D']), 2))}"]
        assert [x for x in l3d if x not in extra] == l3
    e7 = [json.loads(l) for l in open(ROOT / 'data' / 'free_form.jsonl', encoding='utf-8')]
    assert len(e7) == 204 and len({r['G']['text'] for r in e7}) == 204
    for r in e7:
        src = test[r['source']]
        assert all(r['G'][k] in r['G']['text'] for k in ('object', 'from', 'to'))
        assert (r['K'], r['D'], r['A']) == (src['K'], src['D'], src['A'])
        assert verify(dict(r, G=src['G'])) is None             # free-form wording is not a template by design; the rest checks
    # the LLMLingua-2 prompts cover test, train and val (llmlingua2 trains on them), one line per row
    lingua = [json.loads(l)['id'] for l in open(ROOT / 'data' / 'llmlingua2.jsonl', encoding='utf-8')]
    rows = {s: [json.loads(l)['id'] for l in open(ROOT / 'data' / f'{s}.jsonl', encoding='utf-8')] for s in ('test', 'train', 'val')}
    assert len(lingua) == len(set(lingua)) and set(lingua) == {i for ids in rows.values() for i in ids}
