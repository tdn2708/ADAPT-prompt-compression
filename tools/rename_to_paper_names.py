"""Rename the data files and the run files from the working codes used during the experiments to the names of the paper.

    python tools/rename_to_paper_names.py            # dry run: print every move
    python tools/rename_to_paper_names.py --apply    # do it (once; it refuses to overwrite anything)

The experiments ran with short working codes (configuration P2d, prompt level L3, split e1_near, folder phi3_r4_s1, ...).
The paper uses readable names, so the run files and data were renamed once with this script, after the last run. Only
names change: every number, prediction and generation stays as it was. Inside a run file the script maps the configuration
name, the prompt levels (fields level, final_level, cross, bench_extra, the keys of bench and of results/predictions such
as 'test@L3'), the split names (field sets, result keys, the split part of every row id). The fields out and data keep the
paths of the machine that made the run, as a record of where it ran. Runs that the paper does not report are kept, under
results/exploratory/. The tables below are the full mapping, old -> new.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL = {'q05': 'qwen2.5-0.5b', 'q15': 'qwen2.5-1.5b', 'q3b': 'qwen2.5-3b', 'q7b': 'qwen2.5-7b', 'l1b': 'llama-3.2-1b',
         'l3b': 'llama-3.2-3b', 'phi3': 'phi-3-mini'}
CONFIG = {'P1': 'full', 'P2': 'compressed', 'P3lin': 'progressive', 'P3': 'progressive_4_levels', 'P2d': 'compressed_code',
          'P1d': 'full_code', 'C1': 'step_by_step', 'C2': 'step_by_step_compressed', 'P2lingua': 'llmlingua2',
          'ZS': 'zero_shot', 'ZSC': 'zero_shot_step_by_step', 'EX': 'object_naming', 'P1r4': 'full_rerun',
          'P1x4': 'full_4x_data', 'P2d2400': 'compressed_code_2400_steps', 'P2r': 'compressed_payload_rule',
          'P2m': 'compressed_mass_sentences', 'P2mr': 'compressed_payload_rule_mass_sentences',
          'P3m': 'progressive_mass_sentences', 'P1long': 'full_2400_steps', 'P2clip': 'compressed_grad_clip',
          'C1e6': 'step_by_step_longer_eval', 'C2e6': 'step_by_step_compressed_longer_eval'}
# P3 at 2,400 steps (folder phi3_2400) becomes progressive_4_levels_2400_steps, see FOLDERS
LEVEL = {'L0': 'full', 'L1': 'no_role_question', 'L2': 'no_field_descriptions', 'L3': 'compressed',
         'L3d': 'compressed_code', 'L0d': 'full_code', 'L0lingua': 'llmlingua2', 'L3m': 'compressed_mass_sentences',
         'L3r': 'compressed_payload_rule', 'L3mr': 'compressed_payload_rule_mass_sentences',
         # values of --level that are schedules
         'progressive': 'progressive_4_levels', 'progressive_linear': 'progressive',
         'progressive_m': 'progressive_mass_sentences'}
SPLIT = {'e1_near': 'near_threshold', 'e2_dpair': 'd_pairs', 'e3_kpair': 'k_pairs', 'e4_gpair': 'g_pairs',
         'e5_heldout': 'unseen', 'e6_longer': 'longer_than_train', 'e7_paraphrase': 'free_form'}

# (old folder, old config prefix) -> (new folder under results/, new config name); EXP = not reported in the paper
EXP = 'exploratory/'
FOLDERS = {
    'q05': {'P1': 'full', 'P2': 'compressed', 'P3': 'progressive_4_levels', 'P3lin': 'progressive', 'ZS': 'zero_shot'},
    'q05_p2d': {'P2d': 'compressed_code'},
    'q15': {'P1': 'full', 'P2': 'compressed', 'P3': 'progressive_4_levels', 'P3lin': 'progressive', 'ZS': 'zero_shot',
            'P2r': 'compressed_payload_rule', 'P2m': EXP + 'compressed_mass_sentences',
            'P2mr': EXP + 'compressed_payload_rule_mass_sentences', 'P3m': EXP + 'progressive_mass_sentences'},
    'q15_cot': {'C1': 'step_by_step', 'C2': EXP + 'step_by_step_compressed'},
    'q15_cot_seeds': {'C1': 'step_by_step'},
    'q15_cot_e6': {'C1e6': EXP + 'step_by_step_longer_eval', 'C2e6': EXP + 'step_by_step_compressed_longer_eval'},
    'q15_p2d': {'P2d': 'compressed_code'},
    'q3b': {'P1': EXP + 'full', 'P2': EXP + 'compressed', 'P3': EXP + 'progressive_4_levels', 'P3lin': EXP + 'progressive',
            'ZS': 'zero_shot'},
    'q3b_2400': {'P1': 'full', 'P2': 'compressed', 'P3': 'progressive_4_levels', 'P3lin': 'progressive'},
    'q3b_clip': {'P2clip': EXP + 'compressed_grad_clip'},
    'q7b': {'P1': 'full', 'P2': 'compressed'},
    'q7b_s1': {'P1': 'full', 'P2': 'compressed'},
    'q7b_s2': {'P1': 'full', 'P2': 'compressed'},
    'q7b_zs': {'ZS': EXP + 'zero_shot', 'ZSC': 'zero_shot_step_by_step', 'EX_q7b': 'object_naming',
               'EX_q15': 'qwen2.5-1.5b:object_naming'},
    'l1b_2400': {'P1': 'full', 'P2': 'compressed', 'P3': 'progressive_4_levels', 'P3lin': 'progressive', 'ZS': 'zero_shot'},
    'l3b_2400': {'P1': 'full', 'P2': 'compressed', 'P3': 'progressive_4_levels', 'P3lin': 'progressive', 'ZS': 'zero_shot'},
    'l3b_cot': {'C1': 'step_by_step', 'C2': EXP + 'step_by_step_compressed'},
    'l3b_cot_seeds': {'C1': 'step_by_step'},
    'l3b_r4': {'P2d': 'compressed_code'},
    'phi3': {'P1': 'full', 'P2': 'compressed', 'P3': 'progressive_4_levels', 'P3lin': 'progressive', 'P1d': 'full_code',
             'ZS': 'zero_shot'},
    'phi3_2400': {'P1x4': 'full_4x_data', 'P1long': EXP + 'full_2400_steps', 'P3': EXP + 'progressive_4_levels_2400_steps'},
    'phi3_cot': {'C1': 'step_by_step', 'C2': EXP + 'step_by_step_compressed'},
    'phi3_cot_e6': {'C1e6': EXP + 'step_by_step_longer_eval', 'C2e6': EXP + 'step_by_step_compressed_longer_eval'},
    'phi3_cot_seeds': {'C1': 'step_by_step'},
    'phi3_lingua': {'P2lingua': 'llmlingua2'},
    'phi3_p2d_2400': {'P2d': 'compressed_code_2400_steps'},
    **{f'phi3_r4_s{s}': {'P1': 'full_rerun', 'P2d': 'compressed_code'} for s in (1, 2, 3407, 4, 5)},
}
# CPU timing: the Kaggle session (reported) and the team machine (not reported)
CPU = {'cpu_latency': EXP + 'cpu_latency_team', 'cpu_latency_kaggle': 'cpu_latency'}    # in this order
# reports made from the run files by scripts; they are written again by those scripts after the rename
REGENERATED = ['dashboard', 'RESULTS.md', 'ROUND3.md']


def folder_model(folder):
    return MODEL[folder.split('_')[0]]


def map_list(s, table):
    return ','.join(table.get(x, x) for x in s.split(',')) if s else s


def map_result_key(k):
    """'e1_near' -> 'near_threshold', 'test@L3' -> 'test@compressed'."""
    if '@' in k:
        s, lv = k.split('@')
        return f'{SPLIT.get(s, s)}@{LEVEL[lv]}'
    return SPLIT.get(k, k)


ID_RE = re.compile(r'^(%s)/' % '|'.join(SPLIT))


def map_id(i):
    return ID_RE.sub(lambda m: SPLIT[m.group(1)] + '/', i) if isinstance(i, str) else i


def rewrite_run(d, new_name):
    d = dict(d)
    d['name'] = new_name
    for f in ('level', 'final_level'):
        if d.get(f):
            d[f] = LEVEL[d[f]]
    for f in ('cross', 'bench_extra'):
        if f in d and d[f]:
            d[f] = map_list(d[f], LEVEL)
    if d.get('sets'):
        d['sets'] = map_list(d['sets'], SPLIT)
    if isinstance(d.get('bench'), dict):
        d['bench'] = {LEVEL[k]: v for k, v in d['bench'].items()}
    if isinstance(d.get('results'), dict):
        d['results'] = {map_result_key(k): ({**v, 'level': LEVEL[v['level']]} if 'level' in v else v)
                        for k, v in d['results'].items()}
    for f in ('predictions', 'generations', 'extracted'):
        if isinstance(d.get(f), dict):
            d[f] = {map_result_key(k): [[map_id(x[0])] + list(x[1:]) for x in v] for k, v in d[f].items()}
    return d


def dump(d):
    return json.dumps(d, indent=1)


def plan_results():
    """[(old path, new path, new name or None for a plain copy)]"""
    R = ROOT / 'results'
    moves = []
    for folder, table in FOLDERS.items():
        for f in sorted((R / folder).glob('*.json')):
            m = re.match(r'^(.*?)(_s\d+)?$', f.stem)
            old_cfg, seed = m.group(1), m.group(2) or ''
            new = table[old_cfg]
            model = folder_model(folder)
            if ':' in new:
                model, new = new.split(':')
            sub = ''
            if new.startswith(EXP):
                sub, new = EXP, new[len(EXP):]
            stem = new + (seed if not (old_cfg in ('ZS', 'ZSC') and seed == '_s0') else '')
            moves.append((f, R / f'{sub}{model}' / f'{stem}.json', stem))
        for lg in sorted(p for p in (R / folder).glob('_logs*') if p.is_dir()):
            sub = EXP if folder in ('q3b', 'q3b_clip', 'q15_cot_e6', 'phi3_cot_e6') else ''
            moves.append((lg, R / f'{sub}{folder_model(folder)}' / '_logs' / f'{folder}{lg.name}', None))
    for folder, new in CPU.items():
        for f in sorted((R / folder).glob('*.json')):
            stem = f.stem if f.stem == 'MACHINE' else MODEL[f.stem.split('_')[1]]
            moves.append((f, R / new / f'{stem}.json', None if f.stem == 'MACHINE' else stem))
    return moves


def plan_data():
    D = ROOT / 'data'
    return [(D / f'{old}.jsonl', D / f'{new}.jsonl') for old, new in SPLIT.items()]


# the LLMLingua-2 prompts (tools/make_lingua.py): file name and the two token-count fields of each line
LINGUA = ('lingua_L0.jsonl', 'llmlingua2.jsonl')
LINGUA_FIELDS = {'tokens_L0': 'tokens_full', 'tokens_L3': 'tokens_compressed'}


def rename_lingua():
    """data/lingua_L0.jsonl -> data/llmlingua2.jsonl with the fields renamed; every other byte stays (second step, run with
    --lingua after the main rename)."""
    old, new = ROOT / 'data' / LINGUA[0], ROOT / 'data' / LINGUA[1]
    assert old.exists() and not new.exists()
    out = []
    for line in old.read_text(encoding='utf-8').splitlines(keepends=True):
        nl = line
        for a, b in LINGUA_FIELDS.items():
            assert nl.count(f'"{a}": ') == 1
            nl = nl.replace(f'"{a}": ', f'"{b}": ')
        r0, r1 = json.loads(line), json.loads(nl)
        assert {LINGUA_FIELDS.get(k, k): v for k, v in r0.items()} == r1
        out.append(nl)
    new.write_text(''.join(out), encoding='utf-8', newline='')
    old.unlink()
    print(f'{old.relative_to(ROOT)} -> {new.relative_to(ROOT)}: {len(out)} lines')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--lingua', action='store_true', help='only the LLMLingua-2 file (second step)')
    a = ap.parse_args()
    if a.lingua:
        return rename_lingua()
    moves, data = plan_results(), plan_data()
    targets = [n for _, n, _ in moves] + [n for _, n in data]
    assert len(set(targets)) == len(targets), 'two files would get the same name'
    sources = {o for o, _, _ in moves}
    assert not [n for n in targets if n.exists() and n not in sources], 'a target already exists'
    for i, (o, n, _) in enumerate(moves):                # a target that is also a source must be moved away first
        assert all(j < i for j, (o2, _, _) in enumerate(moves) if o2 == n), f'{n} is overwritten before it moves'
    left = [f for folder in list(FOLDERS) + list(CPU) for f in (ROOT / 'results' / folder).iterdir()
            if f not in {o for o, _, _ in moves}]
    assert not left, f'files without a rule: {left}'
    for old, new, name in moves:
        print(f'{old.relative_to(ROOT)} -> {new.relative_to(ROOT)}')
    for old, new in data:
        print(f'{old.relative_to(ROOT)} -> {new.relative_to(ROOT)}')
    if not a.apply:
        print(len(moves), 'run moves,', len(data), 'data files (dry run)')
        return
    # data: only the split prefix of the id and the split field change
    for old, new in data:
        s_old, s_new = old.stem, new.stem
        out = []
        for line in old.read_text(encoding='utf-8').splitlines(keepends=True):
            nl = line.replace(f'"id": "{s_old}/', f'"id": "{s_new}/', 1).replace(f'"split": "{s_old}"',
                                                                                 f'"split": "{s_new}"', 1)
            r0, r1 = json.loads(line), json.loads(nl)
            assert r1['id'] == map_id(r0['id']) and r1['split'] == s_new
            assert {k: v for k, v in r0.items() if k not in ('id', 'split')} == \
                   {k: v for k, v in r1.items() if k not in ('id', 'split')}
            out.append(nl)
        new.write_text(''.join(out), encoding='utf-8', newline='')
        old.unlink()
    # runs
    for old, new, name in moves:
        new.parent.mkdir(parents=True, exist_ok=True)
        if old.is_dir() or name is None:                 # log folders and MACHINE.json: moved as they are
            shutil.move(str(old), str(new))
            continue
        raw = old.read_bytes().decode('utf-8')
        eol = '\r\n' if '\r\n' in raw else '\n'
        text = raw.replace('\r\n', '\n')
        d = json.loads(text)
        assert dump(d) == text, f'{old}: not in the json.dump(indent=1) format'
        out = dump(rewrite_run(d, name) if name else d)
        new.write_bytes(out.replace('\n', eol).encode('utf-8'))
        old.unlink()
    for folder in list(FOLDERS) + list(CPU):
        p = ROOT / 'results' / folder
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()
    for name in REGENERATED:
        p = ROOT / 'results' / name
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    print('done:', len(moves), 'run files and log folders,', len(data), 'data files')


if __name__ == '__main__':
    main()
