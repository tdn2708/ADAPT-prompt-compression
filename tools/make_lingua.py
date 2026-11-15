"""Run-time compression baseline: LLMLingua-2 compresses the full user text of every test
row to the token count of its compressed text, so the two compressions are compared at the same length.

    python tools/make_lingua.py               -> data/llmlingua2.jsonl (CPU, ~15 min; downloads the LLMLingua-2 model once)
    python tools/make_lingua.py train,val     adds the train and val rows (~45 min), for the fair baseline llmlingua2:
                                              a LoRA trained on LLMLingua-2 prompts

Rows of the other sets already in the file are kept as they are, so adding train and val leaves the test rows unchanged.

Each line: id, the compressed text (read by the prompt level llmlingua2), the rate asked, the Qwen2.5 token counts of the
full text (tokens_full), of the compressed level (tokens_compressed) and of the LLMLingua-2 text (tokens), and how many
of the numbers of the row (every number of D and K that the compressed level keeps) are still in the LLMLingua-2 text.
LLMLingua-2 is used as published (microsoft/llmlingua-2-xlm-roberta-large-meetingbank); only the line breaks are forced,
so the request line and the field lines stay apart.
"""

import json
import re
import sys
import time
from pathlib import Path

from llmlingua import PromptCompressor
from transformers import AutoTokenizer

from adapt.prompts import user_text

ROOT = Path(__file__).resolve().parent.parent
NUM = re.compile(r'\d+(?:\.\d+)?')


def main():
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    n = lambda t: len(tok(t, add_special_tokens=False)['input_ids'])
    comp = PromptCompressor(model_name='microsoft/llmlingua-2-xlm-roberta-large-meetingbank', use_llmlingua2=True,
                            device_map='cpu')
    sets = (sys.argv[1] if len(sys.argv) > 1 else 'test').split(',')
    rows = [json.loads(l) for s in sets for l in open(ROOT / 'data' / f'{s}.jsonl', encoding='utf-8')]
    path = ROOT / 'data' / 'llmlingua2.jsonl'
    keep = [json.loads(l) for l in open(path, encoding='utf-8')] if path.exists() else []
    keep = [d for d in keep if d['id'].split('/')[0] not in sets]
    out, t0 = [], time.time()
    for i, r in enumerate(rows):
        l0, l3 = user_text(r, 'full'), user_text(r, 'compressed')
        rate = n(l3) / n(l0)
        text = comp.compress_prompt(l0, rate=rate, force_tokens=['\n'], drop_consecutive=True)['compressed_prompt']
        need = NUM.findall(l3)
        have = NUM.findall(text)
        kept = sum(min(need.count(x), have.count(x)) for x in set(need))
        out.append({'id': r['id'], 'text': text, 'rate': round(rate, 4), 'tokens_full': n(l0), 'tokens_compressed': n(l3),
                    'tokens': n(text), 'numbers': len(need), 'numbers_kept': kept})
        if (i + 1) % 100 == 0:
            print(f'{i + 1}/{len(rows)} {time.time() - t0:.0f}s', flush=True)
    with open(path, 'w', encoding='utf-8') as fh:
        for d in keep + out:
            fh.write(json.dumps(d, ensure_ascii=False) + '\n')
    tk = sum(d['tokens'] for d in out) / len(out)
    t3 = sum(d['tokens_compressed'] for d in out) / len(out)
    allk = sum(d['numbers_kept'] == d['numbers'] for d in out)
    print(f'tokens: LLMLingua-2 {tk:.0f}, compressed level {t3:.0f}; rows with every number kept: {allk}/{len(out)}')


if __name__ == '__main__':
    main()
