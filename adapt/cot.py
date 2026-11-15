"""Step-by-step answers (H8): the rules written out as a short worked check, then the answer, so a model can show each
step instead of answering with one token. The text is computed from the row by the rules, never written by a language
model; check_rationale() recomputes every line from the row.

    payload: alarm clock 0.47 <= 1.38 ok
    reach_pick: 0.56 <= 0.68 ok
    reach_place: 0.47 <= 0.68 ok
    height: 1.89 > 1.79 violated
    clearance: min(2.27, 0.39, 0.56, 1.03) = 0.39 >= 0.26 ok
    battery: 0.46 * 49.5 = 22.77, 22.77 + 28 = 50.77 <= 64 ok
    Feasible: False
"""

import re

from .problem import RULES, fmt, g

COT_QUESTION = 'Check every limit step by step, then answer with Feasible: True or Feasible: False.'
ANSWER_RE = re.compile(r'Feasible:\s*(True|False)')


def _r2(x):
    return f'{x:.2f}'.rstrip('0').rstrip('.')


def rationale(row):
    G, K, D = row['G'], row['K'], row['D']
    t, v, o = K['limits'], g(G, K, D), G['object']
    ok = {r: 'ok' if v[r] >= -1e-9 else 'violated' for r in RULES}
    le = lambda r: '<=' if v[r] >= -1e-9 else '>'
    ge = lambda r: '>=' if v[r] >= -1e-9 else '<'
    trip = t['energy'] * D['route_length']
    lines = [f"payload: {o} {fmt(K['masses'][o])} {le('payload')} {fmt(t['payload'])} {ok['payload']}",
             f"reach_pick: {fmt(D['reach_pick'])} {le('reach_pick')} {fmt(t['reach'])} {ok['reach_pick']}",
             f"reach_place: {fmt(D['reach_place'])} {le('reach_place')} {fmt(t['reach'])} {ok['reach_place']}",
             f"height: {fmt(D['place_height'])} {le('height')} {fmt(t['height'])} {ok['height']}",
             f"clearance: min({', '.join(fmt(c) for c in D['clearance'])}) = {fmt(min(D['clearance']))} "
             f"{ge('clearance')} {fmt(t['clearance'])} {ok['clearance']}",
             f"battery: {fmt(t['energy'])} * {fmt(D['route_length'])} = {_r2(trip)}, {_r2(trip)} + {fmt(t['reserve'])} = "
             f"{_r2(trip + t['reserve'])} {le('battery')} {fmt(D['battery'])} {ok['battery']}",
             f"Feasible: {not any(x == 'violated' for x in ok.values())}"]
    return '\n'.join(lines)


def parse_answer(text):
    """The last 'Feasible: True/False' of a generated text, or None if there is none."""
    m = ANSWER_RE.findall(text)
    return None if not m else m[-1] == 'True'


def check_rationale(row):
    """Every displayed comparison agrees with the rules, and the last line is the answer of the row."""
    text = rationale(row)
    want = bool(row['A']['feasible'])
    assert parse_answer(text) is want, (row['id'], text)
    bad = [ln.split(':')[0] for ln in text.split('\n')[:-1] if ln.endswith('violated')]
    assert bad == list(row['A']['violated']), (row['id'], bad, row['A']['violated'])
    # the rounded battery numbers shown still order the same way as the exact ones (every |slack| >= 1%)
    tr = text.split('\n')[5].split(' ')
    shown_need, sign, battery = float(tr[-4]), tr[-3], float(tr[-2])
    assert (shown_need <= battery) == (sign == '<='), (row['id'], text)
    return True
