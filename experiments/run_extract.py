"""Baseline "Object naming" of the paper: a language model only reads the request and names the object to move; code
then checks the six rules with that object. In this problem the object is the one thing the rules need from the words of
G (D and the limits are numbers already), so this is the pipeline "language model extracts the field, code checks".

    python -m experiments.run_extract --name object_naming --model Qwen/Qwen2.5-1.5B-Instruct --sets test,free_form

Zero-shot, greedy, at most --max-new tokens. The answer is the catalog name that appears first in the generation; a
generation with no catalog name counts as wrong (pred = not label) and in 'unparsed'. Writes <out>/<name>.json in the
format of run.py (predictions [id, pred, margin], margin +1 / -1), plus the extracted objects.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from adapt.prompts import TEMPLATE_KW  # noqa: E402
from adapt.problem import violated  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
ap.add_argument('--load-4bit', action='store_true')
ap.add_argument('--sets', default='test,free_form')
ap.add_argument('--eval-batch', type=int, default=16)
ap.add_argument('--max-new', type=int, default=16)
ap.add_argument('--data', default=str(ROOT / 'data'))
ap.add_argument('--out', default=str(ROOT / 'results' / 'runs'))
a = ap.parse_args()
OUT = Path(a.out) / f'{a.name}.json'
if OUT.exists():
    print(f'{OUT} exists, skipped')
    raise SystemExit(0)
OUT.parent.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP = torch.bfloat16 if DEV == 'cuda' and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
if a.load_4bit:
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map={'': 0}, quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=AMP, bnb_4bit_use_double_quant=True))
else:
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=AMP if DEV == 'cuda' else torch.float32).to(DEV)
model.eval()
log = vars(a) | {'device': DEV, 'mode': 'extract', 'results': {}, 'predictions': {}, 'extracted': {}}


def prompt(row):
    names = ', '.join(row['K']['masses'])
    user = (f"Request: {row['G']['text']}\nObjects in the room: {names}.\n"
            'Which one of these objects does the person want the robot to move? Answer with its name only.')
    return tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True,
                                   **TEMPLATE_KW)


def first_name(text, names):
    t = text.lower()
    hits = sorted((t.find(n.lower()), -len(n), n) for n in names if n.lower() in t)
    return hits[0][2] if hits else None


label = lambda r: bool(r['A']['feasible'])
for name in [s for s in a.sets.split(',') if s]:
    rows = [json.loads(l) for l in open(Path(a.data) / f'{name}.jsonl', encoding='utf-8')]
    t0, outs = time.time(), []
    with torch.no_grad():
        for i in range(0, len(rows), a.eval_batch):
            x = tok([prompt(r) for r in rows[i:i + a.eval_batch]], add_special_tokens=False, padding=True,
                    return_tensors='pt').to(DEV)
            y = model.generate(**x, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            outs += [tok.decode(s[x['input_ids'].shape[1]:], skip_special_tokens=True) for s in y]
    got = [first_name(o, r['K']['masses']) for o, r in zip(outs, rows)]
    pred = [(not violated(dict(r['G'], object=g), r['K'], r['D'])) if g else not label(r) for g, r in zip(got, rows)]
    yy, pp = np.array([label(r) for r in rows]), np.array(pred)
    bal = 100 * float(np.mean([np.mean(pp[yy == k] == k) for k in (True, False)]))
    res = {'n': len(rows), 'balanced': bal, 'object_acc': 100 * float(np.mean([g == r['G']['object'] for g, r in zip(got, rows)])),
           'unparsed': int(sum(g is None for g in got)), 'sec': time.time() - t0}
    log['results'][name] = res
    log['predictions'][name] = [[r['id'], bool(p), 1.0 if p else -1.0] for r, p in zip(rows, pred)]
    log['extracted'][name] = [[r['id'], g, o] for r, g, o in zip(rows, got, outs)]
    print(f"[{name}] n={res['n']} balanced {bal:.1f} object {res['object_acc']:.1f}% unparsed {res['unparsed']}", flush=True)
json.dump(log, open(OUT, 'w'), indent=1)
print('saved', OUT)
