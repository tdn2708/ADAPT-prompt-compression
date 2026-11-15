"""H8: does writing the steps out lift the boundary? Train a LoRA to answer with the worked check of adapt/cot.py (the six
rules, then 'Feasible: True/False') instead of one token, then score by greedy generation.

    python -m experiments.run_cot --name step_by_step_s3407 --model unsloth/Phi-3-mini-4k-instruct --level full \
        --steps 1200
    python -m experiments.run_cot --name step_by_step_compressed_s3407 ... --level compressed   # steps written out
    python -m experiments.run_cot --name zero_shot_step_by_step --model Qwen/Qwen2.5-7B-Instruct --load-4bit --steps 0 \
        --max-new 512                                                     # zero-shot, step by step (round 4)

Same recipe as experiments/run.py (LoRA r16 on all projections, lr 2e-4, 5 warm-up steps then linear decay, batch 8,
best-on-val weights); the loss is on the answer text only (the prompt tokens are masked). A row whose generation has no
'Feasible: True/False' counts as wrong (pred = not label) and is counted in 'unparsed'. Writes <out>/<name>.json in the
format of run.py (predictions [id, pred, margin], margin +1 / -1), plus the generations and the time per decision.
"""

import argparse
import gc
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from adapt.cot import COT_QUESTION, parse_answer, rationale  # noqa: E402
from adapt.prompts import QUESTION, TEMPLATE_KW, user_text  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SETS = ['test', 'near_threshold', 'd_pairs', 'k_pairs', 'g_pairs', 'unseen', 'longer_than_train']
ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--model', default='unsloth/Phi-3-mini-4k-instruct')
ap.add_argument('--level', default='full', choices=['full', 'compressed'])
ap.add_argument('--steps', type=int, default=1200, help='0: no training (zero-shot)')
ap.add_argument('--load-4bit', action='store_true', help='4-bit NF4 weights (bitsandbytes), for the 7B reference')
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--micro-batch', type=int, default=4)
ap.add_argument('--eval-batch', type=int, default=16)
ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--clip', type=float, default=1.0, help='gradient norm clipping (0: off)')
ap.add_argument('--seed', type=int, default=3407)
ap.add_argument('--lora-targets', default='all-linear')
ap.add_argument('--val-every', type=int, default=300)
ap.add_argument('--val-n', type=int, default=120)
ap.add_argument('--max-new', type=int, default=200)
ap.add_argument('--sets', default=','.join(SETS))
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--bench', type=int, default=20, help='time this many test prompts at batch 1 (generation)')
ap.add_argument('--data', default=str(ROOT / 'data'))
ap.add_argument('--out', default=str(ROOT / 'results' / 'runs'))
a = ap.parse_args()
OUT = Path(a.out) / f'{a.name}.json'
if OUT.exists():
    print(f'{OUT} exists, skipped')
    raise SystemExit(0)
OUT.parent.mkdir(parents=True, exist_ok=True)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP = torch.bfloat16 if DEV == 'cuda' and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def prompt(row):
    """Chat text up to the assistant turn; at full the one-token question is replaced by the step-by-step one."""
    user = user_text(row, a.level)
    user = user.replace(QUESTION, COT_QUESTION) if QUESTION in user else user + '\n' + COT_QUESTION
    return tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True,
                                   **TEMPLATE_KW)


def load(name):
    return [json.loads(l) for l in open(Path(a.data) / f'{name}.jsonl', encoding='utf-8')]


label = lambda r: bool(r['A']['feasible'])


def balanced(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return 100 * float(np.mean([np.mean(p[y == k] == k) for k in (True, False) if np.any(y == k)]))


def sync():
    if DEV == 'cuda':
        torch.cuda.synchronize()


LOAD_DTYPE = AMP if DEV == 'cuda' else torch.float32
if a.load_4bit:
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map={'': 0}, quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=AMP, bnb_4bit_use_double_quant=True))
else:
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=LOAD_DTYPE).to(DEV)
log = vars(a) | {'device': DEV, 'amp': str(AMP), 'torch': torch.__version__, 'mode': 'cot'}
if DEV == 'cuda':
    log['gpu'] = torch.cuda.get_device_name(0)


def generate(rows, batch=None):
    """Greedy generations for rows (left padding), the parsed answers and the number of new tokens."""
    tok.padding_side = 'left'
    outs, n_new = [], []
    bs = batch or a.eval_batch
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            x = tok([prompt(r) for r in rows[i:i + bs]], add_special_tokens=False, padding=True, return_tensors='pt').to(DEV)
            with torch.autocast(DEV, dtype=AMP, enabled=DEV == 'cuda'):
                y = model.generate(**x, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                                   use_cache=True)
            for seq in y[:, x['input_ids'].shape[1]:]:
                ids = [t for t in seq.tolist() if t != tok.pad_token_id]
                n_new.append(len(ids))
                outs.append(tok.decode(ids, skip_special_tokens=True))
    sync()
    return outs, [parse_answer(o) for o in outs], n_new


# ---- training ----------------------------------------------------------------------------------------------------------
if a.steps:                                       # --steps 0: zero-shot, the base model answers as it is
    from peft import LoraConfig, get_peft_model  # noqa: E402
    import peft.import_utils as peft_iu  # noqa: E402
    try:
        peft_iu.is_torchao_available()
    except ImportError as e:
        print(f'[peft] {e}; treating torchao as not installed', flush=True)
        import peft.tuners.lora.torchao as peft_tao
        peft_iu.is_torchao_available = peft_tao.is_torchao_available = lambda: False
    rows = load('train')
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    targets = a.lora_targets if a.lora_targets == 'all-linear' else a.lora_targets.split(',')
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, task_type='CAUSAL_LM', target_modules=targets))
    log['lora_trainable_params'] = sum(q.numel() for q in model.parameters() if q.requires_grad)
    for p_ in model.parameters():
        if p_.requires_grad:
            p_.data = p_.data.float()
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.0, 1 - s / a.steps))
    scaler = torch.amp.GradScaler('cuda', enabled=DEV == 'cuda' and AMP == torch.float16)

    by = defaultdict(list)
    for r in load('val'):
        by[(label(r), r['key_rule'])].append(r)
    k = max(1, a.val_n // len(by))
    val_rows = [r for key in sorted(by, key=str) for r in by[key][:k]]
    val_y = [label(r) for r in val_rows]
    val_curve, best, val_sec = [], None, 0.0


    def encode(batch):
        """Right-padded prompt + answer, labels -100 on the prompt and the padding."""
        tok.padding_side = 'right'
        ids, labels = [], []
        for r in batch:
            p = tok(prompt(r), add_special_tokens=False)['input_ids']
            t = tok(rationale(r) + tok.eos_token, add_special_tokens=False)['input_ids']
            ids.append(p + t); labels.append([-100] * len(p) + t)
        L = max(map(len, ids))
        pad = tok.pad_token_id
        x = torch.tensor([s + [pad] * (L - len(s)) for s in ids], device=DEV)
        m = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in ids], device=DEV)
        y = torch.tensor([s + [-100] * (L - len(s)) for s in labels], device=DEV)
        return x, m, y, L


    def val_check(step_done):
        global best
        t_v = time.time()
        model.eval()
        if DEV == 'cuda':
            torch.cuda.empty_cache()
        _, ans, _ = generate(val_rows)
        model.train()
        pred = [p if p is not None else not y for p, y in zip(ans, val_y)]
        bal = balanced(val_y, pred)
        val_curve.append({'step': step_done, 'val_balanced': bal, 'unparsed': sum(p is None for p in ans)})
        if best is None or bal > best['val_balanced']:
            best = dict(val_curve[-1], state={n: q.detach().to('cpu', copy=True) for n, q in model.named_parameters()
                                              if q.requires_grad})
        print(f'[val] step {step_done} balanced {bal:.1f} unparsed {val_curve[-1]["unparsed"]} '
              f'(best {best["val_balanced"]:.1f} at step {best["step"]})', flush=True)
        return time.time() - t_v


    order, losses = [], []
    model.train(); t0 = time.time()
    for step in range(a.steps):
        if len(order) < a.batch:
            order += random.sample(range(len(rows)), len(rows))
        batch = [rows[order.pop()] for _ in range(a.batch)]
        loss_sum, L = 0.0, 0
        for j in range(0, a.batch, a.micro_batch):
            x, m, y, Lj = encode(batch[j:j + a.micro_batch])
            with torch.autocast(DEV, dtype=AMP, enabled=DEV == 'cuda'):
                loss = model(input_ids=x, attention_mask=m, labels=y).loss * (len(x) / a.batch)
            scaler.scale(loss).backward()
            loss_sum += loss.item(); L = max(L, Lj)
        if a.clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, a.clip)
        scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
        losses.append(loss_sum)
        if len(losses) >= 5 and not np.any(np.isfinite(losses[-5:])):
            raise RuntimeError(f'loss not finite for 5 steps in a row at step {step + 1}')
        if (step + 1) % 10 == 0 or step == a.steps - 1:
            print(f'step {step + 1}/{a.steps} loss(10) {np.mean(losses[-10:]):.3f} len {L} {time.time() - t0:.0f}s', flush=True)
        if a.val_every and ((step + 1) % a.val_every == 0 or step == a.steps - 1):
            sync()
            val_sec += val_check(step + 1)
    sync()
    log |= {'train_rows': len(rows), 'steps_run': a.steps, 'train_sec': time.time() - t0 - val_sec,
            'loss_curve': [float(np.mean(losses[i:i + 10])) for i in range(0, len(losses), 10)],
            'val_curve': val_curve, 'val_sec': val_sec, 'best_step': best['step'], 'best_val_balanced': best['val_balanced']}
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, q in best['state'].items():
            params[n].data.copy_(q.to(params[n].device))
    print(f"[val] using step {best['step']} (val balanced {best['val_balanced']:.1f})", flush=True)
    del opt, sched, scaler
    gc.collect()
    if DEV == 'cuda':
        torch.cuda.empty_cache()
    model = model.merge_and_unload()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
model.config.use_cache = True
model.eval()

# ---- scoring -----------------------------------------------------------------------------------------------------------
log['results'], log['predictions'], log['generations'] = {}, {}, {}


def subset(rs):
    if not a.limit or a.limit >= len(rs):
        return rs
    if 'pair_id' in rs[0]:
        ids = sorted({r['pair_id'] for r in rs})[::max(1, len(rs) // a.limit)]
        return [r for r in rs if r['pair_id'] in set(ids)]
    return rs[::max(1, len(rs) // a.limit)]


for name in [s for s in a.sets.split(',') if s]:
    rs = subset(load(name))
    t1 = time.time()
    outs, ans, n_new = generate(rs)
    y = np.array([label(r) for r in rs])
    p = np.array([q if q is not None else not yy for q, yy in zip(ans, y)])
    res = {'level': a.level, 'n': len(rs), 'acc': 100 * float(np.mean(p == y)), 'balanced': balanced(y, p),
           'pred_true_rate': 100 * float(p.mean()), 'unparsed': int(sum(q is None for q in ans)),
           'new_tokens_mean': float(np.mean(n_new)), 'sec': time.time() - t1}
    if 'pair_id' in rs[0]:
        ok = defaultdict(list)
        for r, pp in zip(rs, p):
            ok[r['pair_id']].append(bool(pp) == label(r))
        res['pairs_both_right'] = 100 * float(np.mean([all(v) for v in ok.values()]))
    log['results'][name] = res
    log['predictions'][name] = [[r['id'], bool(pp), 1.0 if pp else -1.0] for r, pp in zip(rs, p)]
    log['generations'][name] = [[r['id'], o] for r, o in zip(rs, outs)]
    print(f"[{name}] {a.level} n={res['n']} balanced {res['balanced']:.1f} unparsed {res['unparsed']} "
          f"new tokens {res['new_tokens_mean']:.0f}" + (f" pairs {res['pairs_both_right']:.1f}%" if 'pairs_both_right' in res else ''),
          flush=True)

# ---- time per decision at batch 1: the prompt and every generated step ------------------------------------------------------
if a.bench:
    test = load('test')
    rs = test[::max(1, len(test) // a.bench)][:a.bench]
    generate(rs[:2], batch=1)                      # warm-up
    ms, new = [], []
    for r in rs:
        sync(); t1 = time.perf_counter()
        _, _, n_new = generate([r], batch=1)
        sync(); ms.append(1000 * (time.perf_counter() - t1)); new += n_new
    log['bench'] = {a.level: {'n': len(rs), 'ms_mean': float(np.mean(ms)), 'ms_median': float(np.median(ms)),
                              'new_tokens_mean': float(np.mean(new)),
                              'prompt_tokens_mean': float(np.mean([len(tok(prompt(r), add_special_tokens=False)['input_ids'])
                                                                   for r in rs]))}}
    print(f"[bench] {a.level} {log['bench'][a.level]['ms_mean']:.0f} ms/decision, "
          f"{log['bench'][a.level]['new_tokens_mean']:.0f} new tokens", flush=True)
json.dump(log, open(OUT, 'w'), indent=1)
print('saved', OUT)
