# Compressing robot decision prompts by information source (ADAPT)

Code, data and run files for the paper *Compressing Robot Decision Prompts by Information Source with the ADAPT
Architecture for Small Language Models* (under review).

A small language model (SLM) on a robot decides whether a pick-and-place request can be carried out. The ADAPT
architecture splits the information of one decision by its source: the **user goal** (G), the **prior world
knowledge** (K), the **dynamic environmental data** from the sensors (D) and the **action** (A). The compression
framework follows from that split, with three rules:

1. knowledge that never changes (the wording of the rules) is learned into the weights with LoRA and removed from the
   prompt;
2. every sentence that carries a number is kept;
3. operations that combine several numbers (the narrowest clearance along a route, the battery a route needs) are
   precomputed by code.

## Layout

```
adapt/                     the problem and the data
  problem.py              six rules, the text of K, the answer A
  generate.py             data generator                       python -m adapt.generate
  verify.py               the single validity check of a row
  prompts.py              prompt levels (full, compressed, + code) and the progressive schedule
  cot.py                  step-by-step rationales (the "Step by step" configuration)
  kaggle.py               two-GPU lanes used to run the experiments on Kaggle
experiments/
  run.py                  fine-tune (LoRA) and score one configuration
  run_cot.py              the same for step-by-step answers
  run_extract.py          the "Object naming" baseline (the model names the object, code checks the rules)
  audit.py, audit_x4.py   independent audit of the data             -> data/AUDIT*.md
  report.py               data statistics, reference points without a language model -> data/REPORT.md
  prompt_tokens.py        tokens per prompt level                    -> data/PROMPTS.md
  export_dashboard.py     tables from the run files                  -> results/dashboard/*.json
  paper_numbers.py        every number of the paper                  -> outputs/numbers.tex
  fig_forest.py, fig_map.py  the result figures                      -> outputs/*.pdf
tools/
  make_lingua.py          LLMLingua-2 compression of the test prompts -> data/llmlingua2.jsonl
  make_paraphrase.py      the free-form requests                      -> data/free_form.jsonl
tests/                    python -m pytest -q
data/                     the dataset (see data/README.md)
results/                  one JSON per run: configuration, curves, scores and every prediction
```

Run files are `results/<model>/<configuration>_s<seed>.json`, with the configuration names of the paper: Full = `full`,
Compressed = `compressed`, Progressive = `progressive`, Compressed + code (proposed) = `compressed_code`, Full + code =
`full_code`, Step by step = `step_by_step`, LLMLingua-2 = `llmlingua2`, Zero-shot = `zero_shot`, Object naming =
`object_naming`. Phi-3-mini's Compressed + code is compared with `full_rerun`, Full trained again in the same batch of
runs.

## Reproduce

```
pip install -r requirements.txt
python -m pytest -q                                  # the problem, the generator, the prompts
python -m adapt.generate                              # rebuilds data/*.jsonl, identical to the files here
python -m experiments.audit
python -m experiments.report
python -m experiments.prompt_tokens                  # needs the Qwen2.5 and Phi-3 tokenizers
python -m experiments.export_dashboard
python -m experiments.paper_numbers                  # every number of the paper from results/ and data/
python experiments/fig_forest.py
python experiments/fig_map.py
```

One run (GPU), for example Phi-3-mini, Compressed + code, seed 5, with the options of
`results/phi-3-mini/compressed_code_s5.json`:

```
python -m experiments.run --name compressed_code_s5 --model unsloth/Phi-3-mini-4k-instruct --level compressed_code \
    --steps 1200 \
    --batch 8 --micro-batch 4 --eval-batch 8 --lora-targets all-linear --seed 5 \
    --sets test,near_threshold,d_pairs,k_pairs,g_pairs,unseen,longer_than_train,free_form --out results/my_runs
```

`python -m experiments.run --help` lists every option; the exact options of every reported run are stored in its JSON
file in `results/`. The runs used Kaggle sessions with two NVIDIA T4 GPUs (PyTorch 2.11.0); Llama-3.2 needs a Hugging
Face account that accepted its license.

## Runs not reported in the paper

Every run we made is kept. The runs the paper reports are in `results/<model>/`; the variants we tried and do not report
are in `results/exploratory/`:

- `exploratory/qwen2.5-1.5b/compressed_mass_sentences_*`, `progressive_mass_sentences_*`,
  `compressed_payload_rule_mass_sentences_*`: compressed prompts that keep the mass sentence, or the mass sentence and the
  payload rule sentence (the paper reports only the one that keeps the payload rule sentence,
  `qwen2.5-1.5b/compressed_payload_rule_*`).
- `exploratory/phi-3-mini/full_2400_steps_s3407.json`, `progressive_4_levels_2400_steps_s3407.json`: Phi-3-mini trained
  for 2,400 steps on the same data.
- `exploratory/qwen2.5-3b/`: Qwen2.5-3B trained for 1,200 steps (the paper uses 2,400 steps), and Compressed with
  gradient clipping (`compressed_grad_clip_*`).
- `exploratory/*/step_by_step_compressed_*`, `*_longer_eval_*`: Step by step on the compressed prompt, and a rerun with
  a longer answer limit to score the longer-than-train set.
- `exploratory/cpu_latency_team/`: CPU timing on the authors' machine; the paper reports the Kaggle CPU
  (`cpu_latency/`).
- `exploratory/qwen2.5-7b/zero_shot.json`: Qwen2.5-7B zero-shot with a one-token answer; the paper reports its
  step-by-step answer.

The Progressive schedule with 4 equal stages (`*/progressive_4_levels_*`) is reported in one sentence of the paper and
stays next to the other runs.

## Names used while the experiments ran

The runs were made with short working codes and renamed once, after the last run, by `tools/rename_to_paper_names.py`.
Only names changed: every number, prediction and generation in the run files is as the run wrote it, and the field `out`
of each run file still holds the folder of the Kaggle session that made it. The full mapping is in that script; the main
names:

| Paper | Run files | Working code |
|---|---|---|
| Full | `full` | `P1` |
| Compressed | `compressed` | `P2` |
| Progressive | `progressive` | `P3lin` |
| Progressive, 4 equal stages | `progressive_4_levels` | `P3` |
| Compressed + code (proposed) | `compressed_code` | `P2d` |
| Full + code | `full_code` | `P1d` |
| Full trained again (baseline of Phi-3-mini's Compressed + code) | `full_rerun` | `P1` in `phi3_r4_s*/` |
| Step by step | `step_by_step` | `C1` |
| LLMLingua-2 | `llmlingua2` | `P2lingua` |
| Zero-shot | `zero_shot` | `ZS` |
| Object naming | `object_naming` | `EX` |
| Prompt levels (Fig. 3 of the paper) | `full`, `no_role_question`, `no_field_descriptions`, `compressed`, `compressed_code`, `full_code`, `llmlingua2` | `L0`, `L1`, `L2`, `L3`, `L3d`, `L0d`, `L0lingua` |
| Test sets | `near_threshold`, `d_pairs`, `k_pairs`, `g_pairs`, `unseen`, `longer_than_train`, `free_form` | `e1_near`, `e2_dpair`, `e3_kpair`, `e4_gpair`, `e5_heldout`, `e6_longer`, `e7_paraphrase` |
| Model folders | `qwen2.5-0.5b`, `qwen2.5-1.5b`, `qwen2.5-3b`, `qwen2.5-7b`, `llama-3.2-1b`, `llama-3.2-3b`, `phi-3-mini` | `q05`, `q15`, `q3b`, `q7b`, `l1b`, `l3b`, `phi3` |

## License

MIT, see `LICENSE`.
