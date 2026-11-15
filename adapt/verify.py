"""The single validity check of a (G, K, D, A) row: verify(row) -> None (valid) or 'part: reason'.

    G  a known object and two different known places; the wording is one of the templates filled with them
    K  every limit present and inside its range; the catalog holds the requested object, every mass inside the range of
       its object; the rule and catalog sentences carry exactly these numbers; context and field meanings unchanged
    D  exactly the six measurements, each of the right type and inside its physical range; the place height inside the
       height range of the destination
    A  equal to the answer the rules give for (G, K, D): feasible, action, violated rules, plan, explanation
Uniqueness of D over the whole set is checked by generate.Registry.
"""

from .problem import (ALL_OBJECTS, ALL_PLACES, CONTEXT, D_FIELDS, D_RANGE, FIELDS, HELD_OUT_TEMPLATES, LIMITS, RANGE,
                      TEMPLATES, make_A, masses_text, rules_text)

TOL = 1e-9


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def inside(x, lo, hi):
    return num(x) and lo - TOL <= x <= hi + TOL


def check_G(G):
    if not isinstance(G, dict) or G.get('object') not in ALL_OBJECTS:
        return 'unknown object'
    if G.get('from') not in ALL_PLACES or G.get('to') not in ALL_PLACES or G['from'] == G['to']:
        return 'unknown or equal places'
    if G.get('text') not in {t.format(o=G['object'], a=G['from'], b=G['to']) for t in TEMPLATES + HELD_OUT_TEMPLATES}:
        return 'wording is not a template filled with this object and these places'
    return None


def check_K(K, G):
    if not isinstance(K, dict) or set(K) != {'context', 'fields', 'rules', 'masses', 'masses_text', 'limits'}:
        return 'missing or extra part'
    t = K['limits']
    if set(t) != set(LIMITS):
        return 'limits missing'
    for k, (lo, hi) in RANGE.items():
        if not inside(t[k], lo, hi):
            return f'limit {k} out of range'
    if not isinstance(t['reserve'], int):
        return 'reserve is not an integer'
    m = K['masses']
    if not isinstance(m, dict) or G['object'] not in m:
        return 'requested object not in the catalog'
    for o, v in m.items():
        if o not in ALL_OBJECTS or not inside(v, 0.6 * ALL_OBJECTS[o], 1.8 * ALL_OBJECTS[o]):
            return f'mass of {o} unknown or out of range'
    if K['rules'] != rules_text(t) or K['masses_text'] != masses_text(m):
        return 'sentences do not carry these numbers'
    if K['context'] != CONTEXT or K['fields'] != FIELDS:
        return 'context or field meanings changed'
    return None


def check_D(D, G):
    if not isinstance(D, dict) or list(D) != D_FIELDS:
        return 'fields differ'
    for k in ('reach_pick', 'reach_place', 'route_length'):
        if not inside(D[k], *D_RANGE[k]):
            return f'{k} out of range'
    if not inside(D['place_height'], *ALL_PLACES[G['to']]):
        return 'place height outside the height range of the destination'
    c = D['clearance']
    if not isinstance(c, list) or not 1 <= len(c) <= 16 or not all(inside(x, *D_RANGE['clearance']) for x in c):
        return 'clearance readings malformed'
    if not isinstance(D['battery'], int) or isinstance(D['battery'], bool) or not inside(D['battery'], *D_RANGE['battery']):
        return 'battery malformed'
    return None


def check_A(A, G, K, D):
    if A != make_A(G, K, D):
        return 'differs from the rules'
    return None


def verify(row):
    for part, check in (('G', lambda: check_G(row.get('G'))), ('K', lambda: check_K(row.get('K'), row['G'])),
                        ('D', lambda: check_D(row.get('D'), row['G'])),
                        ('A', lambda: check_A(row.get('A'), row['G'], row['K'], row['D']))):
        r = check()
        if r:
            return f'{part}: {r}'
    return None
