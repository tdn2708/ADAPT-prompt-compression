"""The problem: can a home service robot carry out a pick-and-place request?

One sample is (G, K, D, A):
    G  goal       the request "move object o from place A to place B", as one wording of a person
    K  knowledge  what is known before acting: the robot's six limits theta, a catalog of object masses, what every
                  field of D means, and the rules written with this sample's limits
    D  data       what the sensors measure for this request: reach at A and at B (arm distance left once the base is
                  parked as close as the furniture allows), height at B, clearance along the route, route length, battery
    A  answer     feasible or not (by the rules), the violated rules, and the plan or the reasons

Six constraints g_i(G, K, D) >= 0, all of them required:
    payload      theta.payload   - mass[o]                                          (G and K: look up the object)
    reach_pick   theta.reach     - D.reach_pick                                     (scalar)
    reach_place  theta.reach     - D.reach_place                                    (scalar, the same limit)
    height       theta.height    - D.place_height                                   (scalar)
    clearance    min(D.clearance) - theta.clearance                                 (minimum over a list)
    battery      D.battery - theta.energy * D.route_length - theta.reserve         (arithmetic)
feasible = every g_i >= 0. The normalized slack s_i = g_i / scale_i (scale = the limit; for the battery, the charge the
trip needs, energy * length + reserve) says how far a sample is from flipping on rule i.
"""

import math

RULES = ['payload', 'reach_pick', 'reach_place', 'height', 'clearance', 'battery']

# ---- K: the robot's limits, drawn per sample ----------------------------------------------------------------------------
# theta ~ N(k0, (0.3 k0)^2) truncated to [0.6, 1.8] k0: every sample is another robot of the same kind.
LIMITS = ['payload', 'reach', 'height', 'clearance', 'energy', 'reserve']
DEFAULT = {'payload': 1.5, 'reach': 0.8, 'height': 1.2, 'clearance': 0.35, 'energy': 0.5, 'reserve': 20}
DIGITS = {'payload': 2, 'reach': 2, 'height': 2, 'clearance': 2, 'energy': 2, 'reserve': 0}
SIGMA, FACTOR = 0.3, (0.6, 1.8)
RANGE = {k: tuple(round(v * f) if DIGITS[k] == 0 else round(v * f, DIGITS[k]) for f in FACTOR) for k, v in DEFAULT.items()}

# ---- K: objects and their typical mass (kg); the catalog of a sample holds masses drawn around these ---------------------
OBJECTS = {'cup': 0.3, 'mug': 0.4, 'alarm clock': 0.4, 'bowl': 0.5, 'plate': 0.6, 'jar of honey': 0.6, 'thermos': 0.7,
           'book': 0.8, 'cutting board': 0.8, 'shoe box': 0.9, 'picture frame': 0.9, 'water bottle': 1.0,
           'bottle of oil': 1.0, 'saucepan': 1.1, 'vase': 1.2, 'kettle': 1.3, 'desk lamp': 1.3, 'frying pan': 1.4,
           'blender jar': 1.5, 'laptop': 1.6, 'bag of apples': 1.8, 'tray of glasses': 1.9, 'watering can': 2.0,
           'jug of milk': 2.1, 'potted plant': 2.2, 'stack of plates': 2.4, 'laundry detergent': 2.5, 'toolbox': 2.6,
           'coffee maker': 2.7, 'stack of books': 2.8, 'pot of soup': 2.9, 'bag of rice': 3.0}
HELD_OUT_OBJECTS = {'candle holder': 0.5, 'tablet': 0.6, 'sewing kit': 0.7, 'speaker': 0.9, 'teapot': 1.1,
                    'bottle of wine': 1.4, 'basket of laundry': 2.0, 'rice cooker': 2.3}
ALL_OBJECTS = {**OBJECTS, **HELD_OUT_OBJECTS}

# ---- G: places; the height of a place (m) bounds the place height D can report ---------------------------------------------
PLACES = {'coffee table': (0.35, 0.50), 'dining table': (0.72, 0.78), 'desk': (0.70, 0.78), 'kitchen counter': (0.88, 0.95),
          'nightstand': (0.50, 0.65), 'sofa': (0.40, 0.48), 'chair': (0.42, 0.48), 'bookshelf': (0.30, 2.00),
          'wall shelf': (1.00, 1.90), 'top of the fridge': (1.70, 1.85), 'windowsill': (0.85, 1.10),
          'cabinet': (0.80, 1.20), 'kitchen island': (0.88, 0.95), 'side table': (0.50, 0.65), 'bed': (0.45, 0.60),
          'bench': (0.40, 0.48), 'ottoman': (0.38, 0.45), 'workbench': (0.85, 0.95), 'bathroom counter': (0.80, 0.90),
          'top of the wardrobe': (1.80, 2.10)}
HELD_OUT_PLACES = {'TV stand': (0.45, 0.60), 'dresser': (0.75, 1.00), 'pantry shelf': (0.30, 1.90),
                   'console table': (0.75, 0.85), 'high shelf': (1.50, 2.10)}
ALL_PLACES = {**PLACES, **HELD_OUT_PLACES}

# ---- D: fixed physical ranges of the measurements ------------------------------------------------------------------------
# reach: the robot drives up to the furniture first, so this is what is left for the arm with the base parked as close
# as the furniture allows (an object deep on a counter against a wall); furniture is at most ~1.2 m deep
D_RANGE = {'reach_pick': (0.10, 1.20), 'reach_place': (0.10, 1.20), 'clearance': (0.10, 2.50),
           'route_length': (3.0, 60.0), 'battery': (5, 100)}
N_READINGS = (4, 8)          # clearance readings along the route, main splits
D_FIELDS = ['reach_pick', 'reach_place', 'place_height', 'clearance', 'route_length', 'battery']


# ---- the rules ----------------------------------------------------------------------------------------------------------
def need(K, D):
    """Battery the trip needs, %: energy per meter times the route length, plus the reserve."""
    t = K['limits']
    return t['energy'] * D['route_length'] + t['reserve']


def g(G, K, D):
    """Constraint values g_i (>= 0: the rule holds), in the units of the quantity."""
    t = K['limits']
    return {'payload': t['payload'] - K['masses'][G['object']],
            'reach_pick': t['reach'] - D['reach_pick'],
            'reach_place': t['reach'] - D['reach_place'],
            'height': t['height'] - D['place_height'],
            'clearance': min(D['clearance']) - t['clearance'],
            'battery': D['battery'] - need(K, D)}


def slack(G, K, D):
    """Normalized slack s_i = g_i / scale_i."""
    t, v = K['limits'], g(G, K, D)
    scale = {'payload': t['payload'], 'reach_pick': t['reach'], 'reach_place': t['reach'], 'height': t['height'],
             'clearance': t['clearance'], 'battery': need(K, D)}
    return {r: v[r] / scale[r] for r in RULES}


EPS = 1e-9                   # float rounding of g (e.g. 0.85 - 0.85); the generator keeps every |s_i| >= 0.01 anyway


def violated(G, K, D):
    v = g(G, K, D)
    return [r for r in RULES if v[r] < -EPS]


def feasible(G, K, D):
    return not violated(G, K, D)


def flip_margin(G, K, D):
    """Smallest relative change of ONE quantity that flips the answer: the smallest slack if feasible, the size of the
    only violation if exactly one rule fails, infinite if two or more fail (no single change flips it)."""
    s, bad = slack(G, K, D), violated(G, K, D)
    if not bad:
        return min(s.values())
    return abs(s[bad[0]]) if len(bad) == 1 else math.inf


# ---- K text -------------------------------------------------------------------------------------------------------------
CONTEXT = ['You are the task planner of a home service robot with a mobile base and one arm.',
           'Decide whether the robot can carry out the request of the person.']
FIELDS = {'reach_pick': 'distance from the arm base to the object, with the base parked as close as the pick-up place allows, m',
          'reach_place': 'distance from the arm base to the drop point, with the base parked as close as the destination allows, m',
          'place_height': 'height of the destination surface, m',
          'clearance': 'distance to the nearest obstacle at points along the route, m',
          'route_length': 'length of the route from the pick-up place to the destination, m',
          'battery': 'battery level, %'}
RULE_TEXT = {'payload': 'The robot can lift objects of at most {payload} kg.',
             'reach_pick': 'The arm can pick up an object at most {reach} m from its base.',
             'reach_place': 'The arm can put an object down at most {reach} m from its base.',
             'height': 'The arm can put an object on a surface at most {height} m high.',
             'clearance': 'The robot needs at least {clearance} m of clearance at every point of its route.',
             'battery': 'Driving uses {energy}% of the battery per meter, and at least {reserve}% must remain at the end '
                        'of the route.'}
ALL_RULES = 'The request can be carried out only if every limit holds.'


def fmt(x):
    return f'{x:g}'


def rules_text(limits):
    return {r: t.format(**{k: fmt(v) for k, v in limits.items()}) for r, t in RULE_TEXT.items()}


def masses_text(masses):
    return 'Known object masses: ' + '; '.join(f'the {o} weighs {fmt(m)} kg' for o, m in masses.items()) + '.'


def make_K(limits, masses):
    return {'context': list(CONTEXT), 'fields': dict(FIELDS), 'rules': rules_text(limits), 'masses': dict(masses),
            'masses_text': masses_text(masses), 'limits': dict(limits)}


# ---- G wordings (written for this dataset): 96 for train/val/test and 32 held-out ones (e5 only), whose verbs and
# phrasings (relocate, fetch, haul, deliver, ferry, shuttle, transport, kindly, ...) never occur in the 96
TEMPLATES = [
    'Move the {o} from the {a} to the {b}.',
    'Take the {o} from the {a} and put it on the {b}.',
    'Please bring the {o} on the {a} over to the {b}.',
    'Can you carry the {o} from the {a} to the {b}?',
    'Put the {o} that is on the {a} onto the {b}.',
    'I need the {o} moved from the {a} to the {b}.',
    'Pick up the {o} from the {a} and place it on the {b}.',
    'Get the {o} off the {a} and set it on the {b}.',
    'Transfer the {o} from the {a} to the {b}.',
    'Could you move the {o} on the {a} to the {b}, please?',
    'Grab the {o} from the {a} and leave it on the {b}.',
    'The {o} on the {a} should go on the {b}.',
    'Bring the {o} from the {a} to the {b}.',
    'Carry the {o} over from the {a} to the {b}.',
    'Please put the {o} from the {a} on the {b}.',
    'Move the {o} off the {a} and onto the {b}.',
    'Would you take the {o} from the {a} to the {b}?',
    'Can you get the {o} from the {a} and put it on the {b}?',
    "I'd like the {o} on the {b} instead of the {a}.",
    "Take the {o} that's on the {a} and move it to the {b}.",
    'Put the {o} on the {b}; it is on the {a} right now.',
    'The {o} is on the {a}. Please move it to the {b}.',
    'Could you bring me the {o} from the {a}? Leave it on the {b}.',
    'Please carry the {o} from the {a} and set it down on the {b}.',
    'Lift the {o} off the {a} and put it on the {b}.',
    'Go to the {a}, pick up the {o}, and put it on the {b}.',
    'From the {a}, take the {o} to the {b}.',
    'On the {b}, please place the {o} from the {a}.',
    'Move the {o} sitting on the {a} over to the {b}.',
    'Can the {o} on the {a} be moved to the {b}?',
    "Put the {o} away on the {b}; it's on the {a}.",
    'Clear the {o} off the {a} and put it on the {b}.',
    "Please move the {o} to the {b}. It's on the {a}.",
    'I left the {o} on the {a}; can you put it on the {b}?',
    'Take the {o} off the {a}, then set it on the {b}.',
    'Could you pick up the {o} on the {a} and put it on the {b}?',
    'Move the {o} from the {a} onto the {b}, please.',
    'Bring the {o} over to the {b} from the {a}.',
    'Set the {o} from the {a} down on the {b}.',
    "I want the {o} that's on the {a} on the {b}.",
    'Please take the {o} to the {b} from the {a}.',
    'Carry the {o} on the {a} to the {b}.',
    'Put the {o} from the {a} over on the {b}.',
    'Can you place the {o} from the {a} on the {b}?',
    'The {o} needs to go from the {a} to the {b}.',
    'Get the {o} from the {a} over to the {b}.',
    'Please pick the {o} up from the {a} and put it on the {b}.',
    'Move the {o} that is sitting on the {a} to the {b}.',
    'Take the {o} and move it from the {a} to the {b}.',
    'Would you put the {o} from the {a} on the {b}, please?',
    'Go get the {o} from the {a} and put it on the {b}.',
    'Bring the {o} on the {a} to the {b}, please.',
    'I need you to move the {o} from the {a} to the {b}.',
    'Take the {o} from the {a} over to the {b}.',
    'The {o} belongs on the {b}, not the {a}. Please move it.',
    'Can you set the {o} from the {a} on the {b}?',
    "Please carry the {o} that's on the {a} to the {b}.",
    'Pick the {o} off the {a} and put it on the {b}.',
    'Lift the {o} from the {a} and place it on the {b}.',
    'Move the {o} across from the {a} to the {b}.',
    'Could you take the {o} on the {a} and leave it on the {b}?',
    "Put the {o} that's on the {a} on the {b}, please.",
    'The {o} on the {a} goes on the {b}.',
    'Take the {o} off the {a} and bring it to the {b}.',
    'Please get the {o} from the {a} and place it on the {b}.',
    'Can you move the {o} over from the {a} to the {b}?',
    'Collect the {o} from the {a} and put it on the {b}.',
    'Bring the {o} from the {a} and set it on the {b}.',
    "I'd like you to carry the {o} from the {a} to the {b}.",
    'Pick up the {o} on the {a} and take it to the {b}.',
    'Move the {o} on the {a} so it is on the {b}.',
    'Take the {o} from the {a}; put it on the {b}.',
    'Please place the {o} that is on the {a} on the {b}.',
    'Could you carry the {o} over to the {b} from the {a}?',
    'Grab the {o} off the {a} and put it on the {b}.',
    'Put the {o} on the {b}, taking it from the {a}.',
    'The {o} should be moved from the {a} to the {b}.',
    "Get the {o} that's on the {a} and leave it on the {b}.",
    'Move the {o} from where it is on the {a} to the {b}.',
    'Can you bring the {o} from the {a} to the {b}, please?',
    'Take the {o} sitting on the {a} to the {b}.',
    'Please lift the {o} off the {a} and set it on the {b}.',
    'Would you carry the {o} on the {a} over to the {b}?',
    'Pick up the {o} at the {a} and put it down on the {b}.',
    'Move the {o} to the {b} from the {a}.',
    'I need the {o} from the {a} put on the {b}.',
    'Bring the {o} that is on the {a} to the {b}.',
    'Take the {o} from the {a} and leave it on the {b}, please.',
    'Could you put the {o} on the {b}? It is on the {a}.',
    'Grab the {o} on the {a} and set it down on the {b}.',
    'Carry the {o} from the {a} and put it on the {b}.',
    "Please move the {o} that's sitting on the {a} onto the {b}.",
    'Pick the {o} up off the {a} and put it on the {b}.',
    'Can you take the {o} that is on the {a} and place it on the {b}?',
    'Put the {o} from the {a} onto the {b}.',
    'Take the {o} over from the {a} to the {b}, please.']
HELD_OUT_TEMPLATES = [
    'Relocate the {o} from the {a} to the {b}.',
    'Would you fetch the {o} from the {a} and drop it on the {b}?',
    'Have the {o} on the {a} brought to the {b}.',
    'Shift the {o} off the {a} and onto the {b}.',
    'Haul the {o} from the {a} over to the {b}.',
    'Deliver the {o} from the {a} to the {b}.',
    'Ferry the {o} from the {a} to the {b}.',
    'Fetch the {o} sitting on the {a} and rest it on the {b}.',
    'Could you relocate the {o} on the {a} to the {b}?',
    'Shift the {o} from the {a} to the {b}, please.',
    'Kindly transport the {o} from the {a} to the {b}.',
    'Transport the {o} on the {a} to the {b}.',
    'Would you mind moving the {o} from the {a} to the {b}?',
    'Do me a favor and get the {o} from the {a} onto the {b}.',
    'Reposition the {o} from the {a} to the {b}.',
    'It would help if the {o} on the {a} ended up on the {b}.',
    'Deliver the {o} that sits on the {a} to the {b}.',
    'Haul the {o} off the {a} and rest it on the {b}.',
    'Fetch the {o} from the {a}, then rest it on the {b}.',
    'Kindly relocate the {o} from the {a} onto the {b}.',
    'Would you be able to transport the {o} from the {a} to the {b}?',
    'Shuttle the {o} from the {a} to the {b}.',
    'Have the {o} moved off the {a} and onto the {b}.',
    'Ferry the {o} sitting on the {a} across to the {b}.',
    "I'd appreciate it if you moved the {o} from the {a} to the {b}.",
    'Reposition the {o} on the {a} so that it rests on the {b}.',
    'Deliver the {o} on the {a} over to the {b}, please.',
    'Fetch the {o} off the {a} and stand it on the {b}.',
    'Could you shuttle the {o} over from the {a} to the {b}?',
    'Haul the {o} that is on the {a} to the {b}.',
    'Kindly shift the {o} on the {a} across to the {b}.',
    'Relocate the {o}: from the {a} to the {b}.']


def make_G(o, a, b, template):
    return {'object': o, 'from': a, 'to': b, 'text': template.format(o=o, a=a, b=b)}


# ---- A ------------------------------------------------------------------------------------------------------------------
def reasons(G, K, D):
    """One sentence per violated rule, with the measured value and the limit."""
    t, o, out = K['limits'], G['object'], []
    for r in violated(G, K, D):
        if r == 'payload':
            out.append(f"the {o} weighs {fmt(K['masses'][o])} kg, more than the payload of {fmt(t['payload'])} kg")
        elif r == 'reach_pick':
            out.append(f"the {o} is {fmt(D['reach_pick'])} m from the arm base, beyond the reach of {fmt(t['reach'])} m")
        elif r == 'reach_place':
            out.append(f"the drop point is {fmt(D['reach_place'])} m from the arm base, beyond the reach of {fmt(t['reach'])} m")
        elif r == 'height':
            out.append(f"the {G['to']} is {fmt(D['place_height'])} m high, above the limit of {fmt(t['height'])} m")
        elif r == 'clearance':
            out.append(f"the route narrows to {fmt(min(D['clearance']))} m, less than the {fmt(t['clearance'])} m needed")
        else:
            out.append(f"the trip needs {need(K, D):.1f}% of battery with the reserve, but only {fmt(D['battery'])}% is left")
    return out


def make_A(G, K, D):
    bad = violated(G, K, D)
    if not bad:
        s = slack(G, K, D)
        tight = min(s, key=s.get)
        return {'feasible': True, 'action': 'Execute', 'violated': [],
                'plan': f"go to the {G['from']}, pick up the {G['object']}, drive to the {G['to']}, put the {G['object']} down",
                'explanation': f'Every limit holds; the tightest is {tight} ({100 * s[tight]:.0f}% to spare).'}
    rs = reasons(G, K, D)
    return {'feasible': False, 'action': 'Refuse', 'violated': bad, 'plan': None,
            'explanation': 'Not feasible: ' + '; '.join(rs) + '.'}
