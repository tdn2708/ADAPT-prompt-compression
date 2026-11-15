"""Prompt levels: every level asks about the same (G, K, D) and is answered with one token (' True' / ' False') after
the assistant prefix 'Feasible:'. Each level drops more wording of K; the request, every number of D, every limit and
every catalog mass stay at every level.

    full                    role and task sentences, D with one sentence per field, the six rules and the catalog in
                            words, the "every limit must hold" sentence, the question
    no_role_question        full without the role, task, "every limit" and question sentences
    no_field_descriptions   no_role_question with D as 'key: value' (the field sentences dropped)
    compressed              no_field_descriptions with the rule and catalog sentences replaced by 'limits: ...' and
                            'masses: ...'

At compressed the model sees only names and numbers: it must know that 'reach 0.85' bounds both reach_pick and
reach_place, that the energy and reserve combine with route_length and battery, and that the request needs every limit.

Control levels, each changing one thing (the four levels above are unchanged):
    compressed_mass_sentences   compressed with the catalog sentence kept ("the kettle weighs 1.3 kg") instead of
                                'masses: ...': does the sentence form carry the look-up?
    full_code                   full plus two lines computed for the model, the smallest clearance reading and the
                                battery the trip needs: is the hard part reading the rule, or the minimum and the
                                arithmetic themselves?
    compressed_payload_rule     compressed plus the payload rule sentence ("The robot can lift objects of at most 1.3 kg.")
    compressed_payload_rule_mass_sentences   compressed plus the payload rule sentence and the catalog sentence
The last three ask which sentence carries the catalog look-up that Qwen2.5-1.5B loses at compressed (it keeps it at
no_field_descriptions).

Round 4:
    compressed_code   compressed plus the two computed lines of full_code as 'key: value': the proposed system,
                      compressed prompt and the aggregates computed by code, still one token
    llmlingua2        the full user text compressed at run time by LLMLingua-2 to about the length of compressed, read
                      from data/llmlingua2.jsonl (tools/make_lingua.py); test, train and val rows (configuration
                      llmlingua2 trains on it)
"""

import json
from pathlib import Path

from .problem import ALL_RULES, D_FIELDS, LIMITS, fmt, need

LEVELS = ['full', 'no_role_question', 'no_field_descriptions', 'compressed']
CONTROL_LEVELS = ['compressed_mass_sentences', 'full_code', 'compressed_payload_rule',
                  'compressed_payload_rule_mass_sentences', 'compressed_code', 'llmlingua2']
LINGUA_FILE = Path(__file__).resolve().parent.parent / 'data' / 'llmlingua2.jsonl'
_LINGUA = {}


def lingua_text(row):
    if not _LINGUA:
        _LINGUA.update((d['id'], d['text']) for d in map(json.loads, open(LINGUA_FILE, encoding='utf-8')))
    return _LINGUA[row['id']]


# schedule progressive_4_levels: a quarter of the steps at each level; progressive_mass_sentences ends at
# compressed_mass_sentences instead
PROGRESSIVE_4_LEVELS = [(0.25, 'full'), (0.50, 'no_role_question'), (0.75, 'no_field_descriptions'),
                        (1.01, 'compressed')]
PROGRESSIVE_MASS_SENTENCES = [(0.25, 'full'), (0.50, 'no_role_question'), (0.75, 'no_field_descriptions'),
                              (1.01, 'compressed_mass_sentences')]
PREFIX = 'Feasible:'
CHOICES = {True: ' True', False: ' False'}
QUESTION = 'Can the robot carry out the request? Answer with True or False only.'


def _v(x):
    return '[' + ', '.join(fmt(v) for v in x) + ']' if isinstance(x, list) else fmt(x)


def user_lines(row, level):
    """The lines of the user text, each tagged with its source: I (role, task, question), G, D, C (precomputed by
    code) or K. user_text joins them; Prompter.segment_tokens counts the tokens of each source (Fig. 4)."""
    G, K, D = row['G'], row['K'], row['D']
    out = []
    if level in ('full', 'full_code'):
        out += [('I', s) for s in K['context']]
    out.append(('G', f"Request: {G['text']}"))
    if level in ('full', 'full_code', 'no_role_question'):
        out += [('D', f"- {K['fields'][k][0].upper()}{K['fields'][k][1:]}: {_v(D[k])}") for k in D_FIELDS]
    else:
        out += [('D', f'{k}: {_v(D[k])}') for k in D_FIELDS]
    if level == 'full_code':
        # every |slack| >= 1% of the need (>= 0.13 points), so 2 decimals never move it across the battery
        out += [('C', f"- Smallest distance to an obstacle along the route, m: {fmt(min(D['clearance']))}"),
                ('C', f"- Battery the trip needs, reserve included, %: {fmt(round(need(K, D), 2))}")]
    if level == 'compressed_code':
        out += [('C', f"clearance_min: {fmt(min(D['clearance']))}"), ('C', f"battery_needed: {fmt(round(need(K, D), 2))}")]
    if level in ('compressed', 'compressed_mass_sentences', 'compressed_payload_rule',
                 'compressed_payload_rule_mass_sentences', 'compressed_code'):
        out.append(('K', 'limits: ' + ', '.join(f"{k} {fmt(K['limits'][k])}" for k in LIMITS)))
        if level in ('compressed_payload_rule', 'compressed_payload_rule_mass_sentences'):
            out.append(('K', K['rules']['payload']))
        out.append(('K', 'masses: ' + ', '.join(f'{o} {fmt(m)}' for o, m in K['masses'].items())
                    if level in ('compressed', 'compressed_payload_rule', 'compressed_code') else K['masses_text']))
    else:
        out.append(('K', K['masses_text']))
        out += [('K', s) for s in K['rules'].values()]
    if level in ('full', 'full_code'):
        out += [('I', ALL_RULES), ('I', QUESTION)]
    return out


def user_text(row, level):
    if level == 'llmlingua2':
        return lingua_text(row)
    return '\n'.join(s for _, s in user_lines(row, level))


def stage(progress, schedule=PROGRESSIVE_4_LEVELS):
    """Level of a quarter schedule (progressive_4_levels) at a training progress in [0, 1]."""
    return next((lv for until, lv in schedule if progress < until), schedule[-1][1])


def stage_linear(progress, rng):
    """Level of one training prompt under the schedule progressive: the expected level falls linearly from full to
    compressed over the first 75% of training, then compressed; between two levels each prompt takes the lower one with
    the remaining share."""
    x = min(len(LEVELS) - 1, progress * (len(LEVELS) - 1) / 0.75)
    i = int(x)
    return LEVELS[min(i + (rng.random() < x - i), len(LEVELS) - 1)]


# Llama-3.x chat templates write today's date into the system header; a fixed date keeps every prompt the same on any
# day (it is the templates' own default). Templates without a date (Qwen2.5, Phi-3) ignore it and render as before.
TEMPLATE_KW = {'date_string': '26 Jul 2024'}


class Prompter:
    """(G, K, D) row + level -> chat text ending with the assistant prefix."""

    def __init__(self, tok):
        self.tok = tok

    def text(self, row, level):
        chat = self.tok.apply_chat_template([{'role': 'user', 'content': user_text(row, level)}],
                                            tokenize=False, add_generation_prompt=True, **TEMPLATE_KW)
        return chat + PREFIX

    def n_tokens(self, row, level):
        return len(self.tok(self.text(row, level), add_special_tokens=False)['input_ids'])

    def segment_tokens(self, row, level):
        """Tokens of the prompt per source, in reading order: [(tag, n), ...] with the tags of user_lines, consecutive
        lines of one source merged, and T (chat template and answer prefix) last. A token belongs to the line where it
        starts; the newline that ends a line belongs to that line. The counts add up to n_tokens."""
        text = self.text(row, level)
        lines = user_lines(row, level)
        start = text.index('\n'.join(s for _, s in lines))
        spans, pos = [], start
        for i, (tag, s) in enumerate(lines):
            hi = pos + len(s) + (i < len(lines) - 1)        # the newline after a line, none after the last one
            spans.append((pos, hi, tag))
            pos = hi
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        out, n_t = [], 0
        for a, _ in enc['offset_mapping']:
            tag = next((t for lo, hi, t in spans if lo <= a < hi), None)
            if tag is None:
                n_t += 1
                continue
            if out and out[-1][0] == tag:
                out[-1][1] += 1
            else:
                out.append([tag, 1])
        return [tuple(x) for x in out] + [('T', n_t)]
