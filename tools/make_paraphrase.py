"""free_form: requests worded freely instead of by the 96 + 32 templates.

    python tools/make_paraphrase.py        -> data/free_form.jsonl

The 204 request frames below were written by a large language model, one sentence structure each, and checked by this
script: every frame names the object {o}, the pick-up place {a} and the destination {b}; no frame repeats a template of
adapt/problem.py; 12 frames also name another object of the catalog ({x}) that must not be moved. Each frame is put on
one test row (17 rows for every (answer, key rule) pair, in id order); only the request text changes, so K, D and the
answer are those of the test row.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from adapt.problem import HELD_OUT_TEMPLATES, TEMPLATES

ROOT = Path(__file__).resolve().parent.parent
FRAMES = [
    'Could you grab the {o} off the {a} and set it down on the {b} for me?',
    'I left the {o} on the {a}; it belongs on the {b}.',
    'The {o} on the {a} should end up on the {b}, please.',
    'Would you mind moving the {o} over from the {a} to the {b}?',
    'Hey, can the {o} go from the {a} to the {b}?',
    "I need the {o} on the {b}. It's sitting on the {a} right now.",
    "Over on the {a} you'll see the {o}; could you bring it to the {b}?",
    'When you get a chance, carry the {o} from the {a} across to the {b}.',
    'It would help me a lot if the {o} were on the {b} instead of the {a}.',
    'Pick up the {o} from the {a}, then place it on the {b}.',
    "The {b} is where I want the {o} that's currently on the {a}.",
    'Please shift the {o} off the {a} and onto the {b}.',
    'Could the {o} be taken from the {a} and left on the {b}?',
    "I'd like you to transfer the {o} from the {a} to the {b}.",
    'Go to the {a}, pick up the {o}, and leave it on the {b}.',
    'Do me a favor and put the {o} from the {a} on the {b}.',
    "Is it possible to get the {o} that's on the {a} onto the {b}?",
    'The {o} is on the {a} but I want it on the {b}; can you handle that?',
    'Bring the {o} over from the {a}, and just set it on the {b}.',
    'Leave the {x} alone and move only the {o} from the {a} to the {b}.',
    'Not the {x}, the {o}: take it from the {a} to the {b}.',
    'Please relocate the {o}, currently on the {a}, to the {b}.',
    'Could you clear the {o} off the {a} by putting it on the {b}?',
    "I'm cleaning the {a}, so the {o} has to go to the {b}.",
    'Can you help me get the {o} from the {a} over onto the {b}?',
    'The {o} needs a new spot: off the {a} and on the {b}.',
    'Take that {o} on the {a} and park it on the {b}, thanks.',
    'Kindly move the {o} lying on the {a} to the {b}.',
    'Before dinner, could you put the {o} from the {a} on the {b}?',
    'Would it be possible for you to carry the {o} from the {a} to the {b}?',
    'See the {o} on the {a}? Put it on the {b}, please.',
    'Get the {o} that I put on the {a} and place it on the {b}.',
    "My {o} is on the {a}; I'd prefer it on the {b}.",
    'Could you swing by the {a}, grab the {o}, and drop it off at the {b}?',
    'I want the {o} moved, from the {a} to the {b}.',
    'Can you take the {o} away from the {a} and set it on the {b}?',
    'Please get the {o} off the {a}; it goes on the {b}.',
    'The {x} can stay, but the {o} on the {a} should go to the {b}.',
    "Move the {o} (it's on the {a}) to the {b}.",
    "If it's no trouble, the {o} sitting on the {a} could use a new home on the {b}.",
    "Could you place the {o} on the {b}? You'll find it on the {a}.",
    "You'll see the {o} on the {a}; it needs to be on the {b}.",
    'Help me tidy up: the {o} goes from the {a} to the {b}.',
    'Can you lift the {o} from the {a} and put it up on the {b}?',
    'Please collect the {o} from the {a} and deliver it to the {b}.',
    'I think the {o} would look better on the {b} than on the {a}; can you move it?',
    "Could you move the {o} to the {b}? It's on the {a} at the moment.",
    'Pick the {o} up off the {a} and put it on the {b} for me.',
    "Let's get the {o} from the {a} onto the {b}.",
    'I need you to bring the {o} sitting on the {a} to the {b}.',
    'Grab the {o}, not the {x}, from the {a} and place it on the {b}.',
    'Can the {o} on the {a} be moved to the {b} now?',
    'Please take the {o} off the {a} and put it somewhere on the {b}.',
    'The {o} should not stay on the {a}; put it on the {b} instead.',
    "Would you carry the {o} over to the {b}? It's on the {a}.",
    'Remove the {o} from the {a} and set it on the {b}, please.',
    'Could you deliver the {o} from the {a} to the {b} when you are free?',
    "I'd like the {o} that is on the {a} placed on the {b}.",
    'Please fetch the {o} lying on the {a} and leave it on the {b}.',
    'Would you be able to put the {o} from the {a} onto the {b}?',
    'Mind grabbing the {o} on the {a} and taking it to the {b}?',
    'Move the {o} on the {a} across to the {b}, if you can.',
    'The {o} from the {a} goes on the {b}.',
    "Can you put the {o} that's resting on the {a} onto the {b}?",
    'Please carry the {o} off the {a}; I want it on the {b}.',
    'I was hoping you could move the {o} from the {a} to the {b}.',
    'First the {a}: pick up the {o} there. Then the {b}: put it down.',
    "It's time to move the {o} from the {a} to the {b}.",
    'Could you pick up the {o} on the {a}? Then set it on the {b}.',
    'Bring the {o} from where it is on the {a} to the {b}.',
    "Place the {o}, which you'll find on the {a}, on the {b}.",
    'We need the {o} off the {a} and on the {b}.',
    'Please, the {o} on the {a}: move it to the {b}.',
    'Can you carry the {o}, not the {x}, from the {a} to the {b}?',
    'Could you set the {o} on the {b}? Get it from the {a} first.',
    'Would you take the {o} sitting on the {a} and put it on the {b}?',
    "I'd love it if the {o} ended up on the {b}; right now it's on the {a}.",
    'Shift the {o} from the {a} over to the {b}, please.',
    'Grab the {o} from the {a} and set it down carefully on the {b}.',
    'The {o} on the {a} is in my way; move it to the {b}.',
    'Could you take the {o} on the {a} to the {b} and leave it there?',
    "Please put the {o} on the {b}; it's currently on the {a}.",
    'I need the {o} from the {a} moved to the {b} soon.',
    'Can you go get the {o} from the {a} and put it on the {b}?',
    'Move the {o} from the {a} to the {b} and leave the {x} where it is.',
    'Would you transfer the {o} on the {a} to the {b} for me?',
    'The {o} belongs on the {b}, not on the {a}; could you fix that?',
    'Please get the {o} off the {a} and onto the {b}.',
    'Could you bring the {o} from the {a} to the {b} before you charge?',
    "Take the {o} that's on the {a} over to the {b}.",
    "I'd like the {o} on the {b}. Could you move it from the {a}?",
    'Can you put the {o} from the {a} down on the {b}?',
    'Please move my {o} from the {a} onto the {b}.',
    'Grab the {o} off the {a} and bring it to the {b}.',
    'Could the {o} on the {a} go onto the {b}?',
    'Kindly bring the {o} from the {a} over to the {b}.',
    'Please carry the {o} from the {a} and place it on the {b}.',
    'Could you see that the {o} on the {a} gets moved to the {b}?',
    'Take the {o} away from the {a}; put it on the {b}.',
    'The {o} is on the {a}. I want it on the {b}. Can you do it?',
    'Can you move the {o} for me? From the {a} to the {b}.',
    'From the {a}, take the {o} and put it on the {b}.',
    "To the {b}, please: the {o} that's on the {a}.",
    "On the {a} you'll find the {o}; please move it to the {b}.",
    'Would you pick up the {o} from the {a} and set it on the {b} for me?',
    "I'd like to have the {o} on the {b}; at the moment it's on the {a}.",
    'Can you get the {o} on the {a} and take it to the {b}? Leave the {x}.',
    "Please put the {o} that's on the {a} on the {b}, thank you.",
    'Could you take the {o} from the {a} and leave it on the {b} for later?',
    "Move the {o} to the {b}; it's on the {a}.",
    "I'd be grateful if you could put the {o} from the {a} on the {b}.",
    'The {o} needs to be taken from the {a} to the {b}.',
    'Can you bring the {o} lying on the {a} over to the {b}?',
    'Get the {o} from the {a} to the {b}, please.',
    'Could you please carry the {o} on the {a} to the {b}?',
    'I want the {o} off the {a}. Put it on the {b}.',
    'Please take the {o} and move it from the {a} to the {b}.',
    "Would you put the {o} on the {b}? It's on the {a} now.",
    "Pick up the {o} that's on the {a} and leave it on the {b}.",
    'Can you take the {o}, the one on the {a}, to the {b}?',
    'Bring the {o} on the {a} to the {b} when you can.',
    'Please move the {o} from the {a}; the {b} is where it should go.',
    'Could you lift the {o} off the {a} and place it on the {b}?',
    'I need the {o} moved off the {a} onto the {b}.',
    "Would you mind putting the {o} that's on the {a} on the {b}?",
    'Carry the {o} from the {a} over to the {b}, thanks.',
    'Can you place the {o} from the {a} on the {b}, please?',
    'Take only the {o} from the {a}, not the {x}, and put it on the {b}.',
    'The {o} should move from the {a} to the {b}.',
    'Could you get the {o} from the {a} and set it on the {b}?',
    'Please bring the {o} over from the {a} to the {b}.',
    "I'd like the {o} to go from the {a} to the {b}.",
    'Can you pick the {o} up from the {a} and drop it on the {b}?',
    'Hey robot, move the {o} from the {a} to the {b}.',
    "Please put the {o} back on the {b}; it's been left on the {a}.",
    'Could you take the {o} that I left on the {a} to the {b}?',
    'Move the {o} that sits on the {a} to the {b}, please.',
    'Would you be so kind as to carry the {o} from the {a} to the {b}?',
    'The {o} on the {a}: could you put it on the {b}?',
    'Can you grab the {o} sitting on the {a} and leave it on the {b}?',
    "I'd like you to pick up the {o} from the {a} and put it on the {b}.",
    'Take the {o} from the {a} and set it on the {b}, not anywhere else.',
    'Please move the {o} on the {a} to the {b} right away.',
    'Could you put the {o}, currently on the {a}, onto the {b}?',
    'Bring the {o} from the {a} and leave it on the {b}.',
    'Can you move the {o} off the {a}? It should be on the {b}.',
    'I need the {o} on the {b}, so please get it from the {a}.',
    'Would you take the {o} from the {a}? Put it on the {b}.',
    'Carry the {o} across from the {a} to the {b}.',
    'Can you put the {o} from the {a} on the {b} and skip the {x}?',
    'Please pick up the {o} from the {a} and bring it to the {b}.',
    'Could you move the {o} lying on the {a} onto the {b}?',
    "The {b} needs the {o} that's on the {a}.",
    'I want you to take the {o} off the {a} and put it on the {b}.',
    'Move the {o} from the {a}, please, and put it on the {b}.',
    'Can you deliver the {o} on the {a} to the {b}?',
    'Please set the {o} on the {b} after taking it from the {a}.',
    'Could you get the {o} off the {a} for me and put it on the {b}?',
    "Would you place the {o} that's on the {a} on the {b}?",
    'Take the {o} lying on the {a} to the {b}.',
    "I'd like the {o} from the {a} brought to the {b}.",
    'Can you bring the {o} from the {a} to the {b} now?',
    'Please grab the {o} from the {a} and set it on the {b}.',
    'Could you shift the {o} on the {a} to the {b}?',
    'Move the {o} from the {a} onto the {b}, but leave the {x}.',
    'The {o} is on the {a}; could you take it to the {b}?',
    "I'd like you to carry the {o} sitting on the {a} to the {b}.",
    'Please collect the {o} on the {a} and put it on the {b}.',
    'Can you take the {o} off the {a} and move it to the {b}?',
    'Would you bring the {o} lying on the {a} to the {b}?',
    'Put the {o} from the {a} over on the {b}, please.',
    'Could you carry the {o} from the {a} and set it down on the {b}?',
    "I need the {o} that's on the {a} to be on the {b}.",
    'Please take the {o} sitting on the {a} over to the {b}.',
    'Can you move the {o} on the {a} so it sits on the {b}?',
    'Would you grab the {o} from the {a} and put it onto the {b}?',
    'Take the {o} off the {a}, then put it on the {b}.',
    "I'd like the {o} on the {a} moved over to the {b}.",
    'Could you pick up the {o} lying on the {a} and set it on the {b}?',
    'Please move the {o} from the {a} to the {b} carefully.',
    'Can you get the {o} sitting on the {a} onto the {b}?',
    'Bring the {o} on the {a} over to the {b}, not the {x}.',
    'Would you take the {o} from the {a} and drop it off on the {b}?',
    'The {o} has to go from the {a} to the {b}; can you do it?',
    'Could you put the {o} lying on the {a} on the {b}?',
    'Please lift the {o} from the {a} and set it on the {b}.',
    "I'd like you to move the {o} sitting on the {a} onto the {b}.",
    'Can you take the {o} from the {a} and put it down on the {b}?',
    "Move the {o} that's on the {a} over to the {b}.",
    'Would you carry the {o} lying on the {a} onto the {b}?',
    'Please get the {o} from the {a} and leave it on the {b}.',
    'Could you bring the {o} sitting on the {a} to the {b}?',
    'I need the {o} on the {a} to be put on the {b}.',
    'Take the {o} from the {a} over to the {b}, please.',
    'Can you move the {o} lying on the {a} to the {b}?',
    'Would you put the {o} sitting on the {a} onto the {b}?',
    'Please carry the {o} on the {a} over to the {b}.',
    'Could you take the {o} lying on the {a} over to the {b}?',
    "I'd like the {o} sitting on the {a} on the {b}.",
    'Grab the {o} on the {a}, leave the {x}, and put the {o} on the {b}.',
    'Can you set the {o} lying on the {a} down on the {b}?',
    'Please take the {o} off the {a} and carry it to the {b}.',
    'Could you move the {o} sitting on the {a} over to the {b}?',
    'Would you bring the {o} from the {a} over onto the {b}?',
]
PER_CELL = 17


def norm(t):
    return re.sub(r'[^a-z{} ]', '', t.lower()).split()


def build():
    assert len(FRAMES) == 12 * PER_CELL, len(FRAMES)
    assert len({tuple(norm(f)) for f in FRAMES}) == len(FRAMES), 'two frames are the same'
    old = {tuple(norm(t)) for t in TEMPLATES + HELD_OUT_TEMPLATES}
    same = [f for f in FRAMES if tuple(norm(f)) in old]
    assert not same, f'frames that repeat a template: {same}'
    for f in FRAMES:
        assert all(k in f for k in ('{o}', '{a}', '{b}')), f
    test = [json.loads(l) for l in open(ROOT / 'data' / 'test.jsonl', encoding='utf-8')]
    cells = defaultdict(list)
    for r in sorted(test, key=lambda r: int(r['id'].split('/')[1])):
        cells[(bool(r['A']['feasible']), r['key_rule'])].append(r)
    picked = [r for key in sorted(cells, key=str) for r in cells[key][:PER_CELL]]
    out = []
    for i, (r, f) in enumerate(zip(picked, FRAMES)):
        G = r['G']
        others = [o for o in r['K']['masses'] if o != G['object']]
        x = others[0] if others else None
        assert '{x}' not in f or x, (r['id'], f)
        text = f.format(o=G['object'], a=G['from'], b=G['to'], x=x)
        text = text[0].upper() + text[1:]
        assert all(G[k] in text for k in ('object', 'from', 'to')), text
        row = dict(r, id=f'free_form/{i}', split='free_form', source=r['id'],
                   G=dict(G, text=text, frame=i, distractor=x if '{x}' in f else None))
        out.append(row)
    path = ROOT / 'data' / 'free_form.jsonl'
    with open(path, 'w', encoding='utf-8') as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    n_x = sum(row['G']['distractor'] is not None for row in out)
    print(f'{path}: {len(out)} rows, {n_x} with a distractor object, '
          f'{sum(not row["A"]["feasible"] for row in out)} infeasible')


if __name__ == '__main__':
    build()
