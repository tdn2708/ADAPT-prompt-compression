"""One run of the compression study: train a LoRA on one prompt level (or a progressive schedule), keep the weights best
on val, then score every evaluation file once and time the levels at batch 1.

    python -m experiments.run --name zero_shot --model Qwen/Qwen2.5-0.5B-Instruct --train none \
        --cross full,no_role_question,no_field_descriptions --bench 50
    python -m experiments.run --name full_s3407 --level full --cross compressed        # no compression
    python -m experiments.run --name compressed_s3407 --level compressed --cross full  # compressed from the start
    python -m experiments.run --name progressive_s3407 --level progressive --cross full
    python -m experiments.run --name compressed_code_s3407 --level compressed_code     # the proposed system
    Others: progressive_4_levels (a quarter of the steps at each level), full_code, compressed_payload_rule; the
    levels are in adapt/prompts.py

The answer is one token: the logit of ' True' against ' False' after the prefix 'Feasible:'; the loss is on that token
only. LoRA r16 on all projections, lr 2e-4, 5 warm-up steps then linear decay, batch 8. Writes <out>/<name>.json with
every prediction (skipped if it exists, so a lane can be re-run).
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

from adapt.prompts import (CHOICES, CONTROL_LEVELS, LEVELS, PREFIX, PROGRESSIVE_4_LEVELS,  # noqa: E402
                           PROGRESSIVE_MASS_SENTENCES, TEMPLATE_KW, Prompter, stage, stage_linear)

ROOT = Path(__file__).resolve().parent.parent
SETS = ['test', 'near_threshold', 'd_pairs', 'k_pairs', 'g_pairs', 'unseen', 'longer_than_train']
ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
ap.add_argument('--train', default='train', help='train | <path>.jsonl | none')
SCHEDULES = ['progressive', 'progressive_4_levels', 'progressive_mass_sentences']
ap.add_argument('--level', default='compressed', choices=LEVELS + CONTROL_LEVELS + SCHEDULES,
                help='progressive: full -> compressed, the mean level falling linearly over 75%% of the steps; '
                     'progressive_4_levels: a quarter of the steps at each level; progressive_mass_sentences: the same '
                     'ending at compressed_mass_sentences')
ap.add_argument('--cross', default='', help='extra levels to score on test (comma list)')
ap.add_argument('--steps', type=int, default=900)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--micro-batch', type=int, default=0, help='gradient accumulation in parts of this size (same update)')
ap.add_argument('--eval-batch', type=int, default=16)
ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--seed', type=int, default=3407)
ap.add_argument('--lora-targets', default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
                help="comma list, or 'all-linear' (Phi-3)")
ap.add_argument('--val-every', type=int, default=150)
ap.add_argument('--val-n', type=int, default=480, help='val rows per check, balanced per (answer, key rule)')
ap.add_argument('--sets', default=','.join(SETS))
ap.add_argument('--limit', type=int, default=0, help='about this many rows per set, spread evenly (smoke tests)')
ap.add_argument('--bench', type=int, default=0, help='time this many test prompts per level at batch 1')
ap.add_argument('--bench-extra', default='', help='levels timed with the four levels (comma list), e.g. '
                                                  'compressed_code,llmlingua2')
ap.add_argument('--data', default=str(ROOT / 'data'))
ap.add_argument('--out', default=str(ROOT / 'results' / 'runs'))
ap.add_argument('--clip', type=float, default=0.0, help='gradient norm clipping (0: off, as in every run before H9)')
ap.add_argument('--load-4bit', action='store_true', help='4-bit NF4 weights (bitsandbytes), for the 7B reference')
a = ap.parse_args()
OUT = Path(a.out) / f'{a.name}.json'
if OUT.exists():
    print(f'{OUT} exists, skipped')
    raise SystemExit(0)
OUT.parent.mkdir(parents=True, exist_ok=True)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
# fp16 + grad scaler on a T4 (no bf16 in hardware); bf16 on Ampere+; fp32 on CPU
AMP = torch.bfloat16 if DEV == 'cuda' and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
P = Prompter(tok)
SCHEDULE = PROGRESSIVE_MASS_SENTENCES if a.level == 'progressive_mass_sentences' else PROGRESSIVE_4_LEVELS
FINAL = SCHEDULE[-1][1] if a.level.startswith('progressive') else a.level


def answer_ids():
    """Ids of ' True' / ' False' as they follow the prefix in a real prompt (sentencepiece tokenizers share the '▁')."""
    chat = tok.apply_chat_template([{'role': 'user', 'content': 'Q'}], tokenize=False, add_generation_prompt=True,
                                   **TEMPLATE_KW)
    base = tok(chat + PREFIX, add_special_tokens=False)['input_ids']
    ids = {}
    for y, word in CHOICES.items():
        full = tok(chat + PREFIX + word, add_special_tokens=False)['input_ids']
        assert full[:len(base)] == base and len(full) == len(base) + 1, (word, full[len(base) - 2:])
        ids[y] = full[-1]
    assert ids[True] != ids[False], ids
    return ids


ID = answer_ids()


def sync():
    if DEV == 'cuda':
        torch.cuda.synchronize()


def last_logits(model, texts):
    x = tok(texts, add_special_tokens=False, padding=True, return_tensors='pt').to(DEV)
    with torch.autocast(DEV, dtype=AMP, enabled=DEV == 'cuda'):
        out = model(**x, logits_to_keep=1).logits[:, -1, :].float()
    return out, x['input_ids'].shape[1]


OOM_SPLITS = [0]


def margins_of(model, texts):
    """Logit margins True - False; on a CUDA out-of-memory error the batch is split in two and retried."""
    try:
        lg, _ = last_logits(model, texts)
        return (lg[:, ID[True]] - lg[:, ID[False]]).cpu().tolist()
    except torch.cuda.OutOfMemoryError:
        if len(texts) == 1:
            raise
        OOM_SPLITS[0] += 1
        torch.cuda.empty_cache()
        h = len(texts) // 2
        return margins_of(model, texts[:h]) + margins_of(model, texts[h:])


def predict(model, rows, level):
    margins, n_tok = [], []
    with torch.no_grad():
        for i in range(0, len(rows), a.eval_batch):
            texts = [P.text(r, level) for r in rows[i:i + a.eval_batch]]
            n_tok += [len(tok(t, add_special_tokens=False)['input_ids']) for t in texts]
            margins += margins_of(model, texts)
    sync()
    return [m > 0 for m in margins], margins, n_tok


def balanced(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return 100 * float(np.mean([np.mean(p[y == k] == k) for k in (True, False) if np.any(y == k)]))


def load(name):
    return [json.loads(l) for l in open(Path(a.data) / f'{name}.jsonl', encoding='utf-8')]


label = lambda r: bool(r['A']['feasible'])
LOAD_DTYPE = AMP if DEV == 'cuda' else torch.float32
if a.load_4bit:
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map={'': 0}, quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=AMP, bnb_4bit_use_double_quant=True))
else:
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=LOAD_DTYPE).to(DEV)
log = vars(a) | {'device': DEV, 'amp': str(AMP), 'final_level': FINAL, 'torch': torch.__version__}
if DEV == 'cuda':
    log['gpu'] = torch.cuda.get_device_name(0)

# ---- training ----------------------------------------------------------------------------------------------------------
if a.train != 'none':
    from peft import LoraConfig, get_peft_model
    import peft.import_utils as peft_iu
    try:
        peft_iu.is_torchao_available()
    except ImportError as e:                         # Kaggle ships a torchao older than peft accepts; LoRA never uses it
        print(f'[peft] {e}; treating torchao as not installed', flush=True)
        import peft.tuners.lora.torchao as peft_tao
        peft_iu.is_torchao_available = peft_tao.is_torchao_available = lambda: False
    rows = load(a.train) if not a.train.endswith('.jsonl') else [json.loads(l) for l in open(a.train, encoding='utf-8')]
    steps = a.steps
    if a.load_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    targets = a.lora_targets if a.lora_targets == 'all-linear' else a.lora_targets.split(',')
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, task_type='CAUSAL_LM',
                                             target_modules=targets))
    log['lora_trainable_params'] = sum(q.numel() for q in model.parameters() if q.requires_grad)
    for p_ in model.parameters():
        if p_.requires_grad:
            p_.data = p_.data.float()                  # LoRA weights in fp32 (needed by the fp16 grad scaler)
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler('cuda', enabled=DEV == 'cuda' and AMP == torch.float16)
    if DEV == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    val_rows, best = [], None
    if a.val_every:
        by = defaultdict(list)
        for r in load('val'):
            by[(label(r), r['key_rule'])].append(r)
        k = max(1, a.val_n // len(by))
        val_rows = [r for key in sorted(by, key=str) for r in by[key][:k]]
        val_y = [label(r) for r in val_rows]
        val_curve, val_sec = [], 0.0

    def val_check(step_done, tokens_done, train_time):
        global best
        t_v = time.time()
        model.eval()
        if DEV == 'cuda':
            torch.cuda.empty_cache()
        p, _, _ = predict(model, val_rows, FINAL)
        model.train()
        bal = balanced(val_y, p)
        val_curve.append({'step': step_done, 'val_balanced': bal, 'train_tokens': tokens_done, 'train_sec': train_time})
        if best is None or bal > best['val_balanced']:      # ties keep the earlier (cheaper) checkpoint
            best = dict(val_curve[-1], state={n: q.detach().to('cpu', copy=True)
                                             for n, q in model.named_parameters() if q.requires_grad})
        print(f'[val] step {step_done} balanced {bal:.1f} (best {best["val_balanced"]:.1f} at step {best["step"]})', flush=True)
        return time.time() - t_v

    order, losses, n_tok = [], [], 0
    lin_rng = random.Random(a.seed + 1)              # levels of progressive; the data order stream is unchanged
    model.train(); t0 = time.time()
    for step in range(steps):
        lv = (stage(step / steps, SCHEDULE) if a.level in ('progressive_4_levels', 'progressive_mass_sentences')
              else a.level)
        if len(order) < a.batch:
            order += random.sample(range(len(rows)), len(rows))   # epochs without replacement
        batch = [rows[order.pop()] for _ in range(a.batch)]
        if a.level == 'progressive':
            lvs = [stage_linear(step / steps, lin_rng) for _ in batch]
            lv = '/'.join(sorted(set(lvs)))
            texts = [P.text(r, x) for r, x in zip(batch, lvs)]
        else:
            texts = [P.text(r, lv) for r in batch]
        n_tok += sum(len(tok(t, add_special_tokens=False)['input_ids']) for t in texts)
        mb, loss_sum, L = a.micro_batch or a.batch, 0.0, 0
        for j in range(0, a.batch, mb):
            logits, Lj = last_logits(model, texts[j:j + mb])
            target = torch.tensor([ID[label(r)] for r in batch[j:j + mb]], device=DEV)
            loss = torch.nn.functional.cross_entropy(logits, target) * (len(target) / a.batch)   # answer token only
            scaler.scale(loss).backward()
            loss_sum += loss.item(); L = max(L, Lj)
        if a.clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p_ for p_ in model.parameters() if p_.requires_grad], a.clip)
        scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
        losses.append(loss_sum)
        if len(losses) >= 5 and not np.any(np.isfinite(losses[-5:])):
            raise RuntimeError(f'loss not finite for 5 steps in a row at step {step + 1}')
        if (step + 1) % 10 == 0 or step == steps - 1:
            print(f'step {step + 1}/{steps} {lv} loss(10) {np.mean(losses[-10:]):.3f} len {L} {time.time() - t0:.0f}s',
                  flush=True)
        if a.val_every and ((step + 1) % a.val_every == 0 or step == steps - 1):
            sync()
            val_sec += val_check(step + 1, n_tok, time.time() - t0 - val_sec)
    sync()
    log |= {'train_rows': len(rows), 'steps_run': steps, 'train_prompt_tokens': n_tok,
            'train_tokens_per_prompt': n_tok / (steps * a.batch),
            'train_sec': time.time() - t0 - (val_sec if a.val_every else 0.0),
            'loss_curve': [float(np.mean(losses[i:i + 10])) for i in range(0, len(losses), 10)]}
    if a.val_every:
        params = dict(model.named_parameters())
        with torch.no_grad():
            for n, q in best['state'].items():                # back to the best-on-val LoRA weights
                params[n].data.copy_(q.to(params[n].device))
        log |= {'val_curve': val_curve, 'val_sec': val_sec, 'val_n': len(val_rows), 'best_step': best['step'],
                'best_val_balanced': best['val_balanced'], 'tokens_to_best': best['train_tokens']}
        print(f"[val] using step {best['step']} (val balanced {best['val_balanced']:.1f})", flush=True)
    if DEV == 'cuda':
        log['train_peak_mem_gb'] = torch.cuda.max_memory_allocated() / 2**30
    del opt, sched, scaler
    gc.collect()
    if DEV == 'cuda':
        torch.cuda.empty_cache()
    if not a.load_4bit:                                  # a 4-bit base keeps the adapter separate
        model = model.merge_and_unload()
model.eval()


# ---- scoring -----------------------------------------------------------------------------------------------------------
def score(rows, level, name):
    t0 = time.time()
    preds, margins, n_tok = predict(model, rows, level)
    y, p = np.array([label(r) for r in rows]), np.array(preds)
    res = {'level': level, 'n': len(rows), 'acc': 100 * float(np.mean(p == y)), 'balanced': balanced(y, p),
           'pred_true_rate': 100 * float(p.mean()), 'tokens_mean': float(np.mean(n_tok)), 'sec': time.time() - t0}
    if rows and 'pair_id' in rows[0]:
        ok = defaultdict(list)
        for r, pp in zip(rows, p):
            ok[r['pair_id']].append(pp == label(r))
        res['pairs_both_right'] = 100 * float(np.mean([all(v) for v in ok.values()]))
    log['results'][name] = res
    log['predictions'][name] = [[r['id'], bool(pp), round(mm, 3)] for r, pp, mm in zip(rows, preds, margins)]
    print(f"[{name}] {level} n={res['n']} balanced {res['balanced']:.1f} tokens {res['tokens_mean']:.0f} "
          f"pred_true {res['pred_true_rate']:.0f}%" + (f" pairs {res['pairs_both_right']:.1f}%" if 'pairs_both_right' in res else ''),
          flush=True)


def subset(rows):
    if not a.limit or a.limit >= len(rows):
        return rows
    if 'pair_id' in rows[0]:                          # keep whole pairs
        ids = sorted({r['pair_id'] for r in rows})[::max(1, len(rows) // a.limit)]
        return [r for r in rows if r['pair_id'] in set(ids)]
    return rows[::max(1, len(rows) // a.limit)]


log['results'], log['predictions'] = {}, {}
test_rows = subset(load('test'))
for name in [s for s in a.sets.split(',') if s]:
    score(test_rows if name == 'test' else subset(load(name)), FINAL, name)
for lv in [s for s in a.cross.split(',') if s and s != FINAL]:
    score(test_rows, lv, f'test@{lv}')


# ---- latency at batch 1 (one forward = the whole one-token answer) -------------------------------------------------------
def bench_levels(n):
    test = load('test')
    rows = test[::max(1, len(test) // n)][:n]
    levels = LEVELS + [s for s in a.bench_extra.split(',') if s]
    texts = {lv: [P.text(r, lv) for r in rows] for lv in levels}
    ms, rng = defaultdict(list), random.Random(0)
    with torch.no_grad():
        for t in texts['full'][:3]:
            last_logits(model, [t])                     # warm-up
        for i in range(len(rows)):
            for lv in rng.sample(levels, len(levels)):   # random order: clock or heat drift hits every level alike
                sync(); t0 = time.perf_counter()
                last_logits(model, [texts[lv][i]])
                sync(); ms[lv].append(1000 * (time.perf_counter() - t0))
    return {lv: {'n': len(rows), 'tokens_mean': float(np.mean([len(tok(t, add_special_tokens=False)['input_ids'])
                                                              for t in texts[lv]])),
                 'ms_mean': float(np.mean(ms[lv])), 'ms_median': float(np.median(ms[lv]))} for lv in levels}


if a.bench:
    log['bench'] = bench_levels(a.bench)
    for lv, b in log['bench'].items():
        print(f"[bench] {lv} tokens {b['tokens_mean']:.0f} {b['ms_mean']:.1f} ms/prompt", flush=True)
log['eval_oom_splits'] = OOM_SPLITS[0]
json.dump(log, open(OUT, 'w'), indent=1)
print('saved', OUT)
