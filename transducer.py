import time
import logging
from typing import FrozenSet, Set, Dict, Tuple, List

from graphs import draw_transducer

logger = logging.getLogger(__name__)
logger.setLevel(3)

SIGMA = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
         'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
         'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0',
         '1', '2', '3', '4', '5', '6', '7', '8', '9']

class State:
    def __init__(self):
        self.index = State.state_counter
        State.state_counter += 1

    def __eq__(self, other):
        return self.index == other.index

    def __lt__(self, other):
        return self.index < other.index

    def __str__(self):
        return f"State({self.index})"
    
    def __repr__(self):
        return f"State({self.index})"

    def __hash__(self):
        return hash(self.index)

State.state_counter = 1

class Signature:
    def __init__(self, is_final: bool, output: str, outgoing: FrozenSet[Tuple[str, str, State]]):
        self.is_final = is_final
        self.output = output
        self.outgoing = outgoing
        
    def __str__(self) -> str:
        return f"Signature({self.is_final}, {self.output}, {self.outgoing})"
    
    def __repr__(self) -> str:
        return f"Signature({self.is_final}, {self.output}, {self.outgoing})"
    
    def __eq__(self, other) -> bool:
        return self.is_final == other.is_final and\
            self.output == other.output and\
            hash(self.outgoing) == hash(other.outgoing)
            
    def __hash__(self) -> int:
        return hash((self.outgoing, self.output, self.is_final))


class Transducer:
    def __init__(self,
                 Q: Set[State],
                 s: State,
                 F: Set[State],
                 deltaT: Dict[Tuple[State, str], State],
                 lambdaT: Dict[Tuple[State, str], str],
                 iota: str,
                 psi: Dict[State, str],
                 itr: Dict[State, int],
                 delta_state_to_chars: Dict[State, List[str]]
                 ):
        self.Q = Q
        self.s = s
        self.F = F
        self.deltaT = deltaT
        self.lambdaT = lambdaT
        self.iota = iota
        self.psi = psi
        self.itr = itr
        self.delta_state_to_chars = delta_state_to_chars


def common_prefix(s1: str, s2: str) -> str:
    n = min(len(s1), len(s2))
    for i in range(n):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[:n]


def remainder_suffix(w: str, s: str) -> str:
    return s[len(w):]


def calc_signature(T: Transducer, q: State) -> Signature:
    is_final = q in T.F
    output = T.psi[q] if is_final else ""
    outgoing = frozenset((c, T.lambdaT[(q, c)], T.deltaT[(q, c)]) for c in T.delta_state_to_chars[q])
    return Signature(is_final, output, outgoing)


def state_seq(deltaT: Dict[Tuple[State, str], State], q: State, w: str) -> List[State]:
    path = [q]
    state = q
    for a in w:
        if (state, a) in deltaT:
            state = deltaT[(state, a)]
            path.append(state)
        else:
            break
        
    return path


# make transducer minimal except for a[0:l-1]
def reduce_except(T: Transducer, a: str, l: int, h: Dict[Signature, State]) -> Tuple[Transducer, Dict[Signature, State]]:
    path = state_seq(T.deltaT, T.s, a)
    
    for p in range(len(path) - 1, l - 0, -1):
        state = path[p]
        
        signature = calc_signature(T, state)
        if signature not in h:
            h[signature] = state
        else:            
            T.Q.remove(state)
            if state in T.F:
                T.F.remove(state)
                del T.psi[state]                  
            
            for s in T.delta_state_to_chars[state]:
                del T.deltaT[(state, s)]
                del T.lambdaT[(state, s)]
                
                
            del T.delta_state_to_chars[state]
            
            T.deltaT[(path[p-1], a[p-1])] = h[signature]
            
            
    # assert set(T.psi.keys()) == T.F, f"\tF: {T.F}\n\tPSI: {T.psi}"
    # assert set(T.delta_state_to_chars.keys()) == T.Q, f"\tQ: {T.Q}\n\tDELTA_CHARS: {T.delta_state_to_chars}"
    # assert set(T.deltaT.keys()) == set(T.lambdaT.keys()), f"\tDELTA: {T.deltaT}\n\tLAMBDA: {T.lambdaT}"
    # assert set(h.keys()) == T.Q, f"Q: {T.Q}\n\th: {h}"

    return T, h
            

def push_output_forward(T: Transducer, t: List[State], v: str, beta: str, k: int) -> Tuple[Dict[Tuple[State, str], str], Dict[State, str]]:
    
    # logger.debug(f"pushing out {(v, beta)}")
    
    L = T.lambdaT
    P = T.psi
    c = ""
    l = ""
    b = beta
    L_i = T.iota
    
    # logger.debug(f"L_i: {L_i}")
    # logger.debug(f"L: {L}")
    # logger.debug(f"P: {P}")
    # logger.debug(f"c: {c}")
    # logger.debug(f"l: {l}")
    # logger.debug(f"b: {b}\n")
    
    c = common_prefix(l + L_i, b)
    l = remainder_suffix(c, l + L_i)
    b = remainder_suffix(c, b)
    
    for s in T.delta_state_to_chars[t[0]]:
        if s != v[0]:
        # L_trans[x_in] = l*x_out
            x_in = (t[0], s)
            x_out = L[(t[0], s)]
            L[x_in] = l + x_out
  

    if t[0] in T.F:
        cacheP = T.psi[t[0]]
        P[t[0]] = l + cacheP
        
    for j in range(k):
        
        # logger.debug(f"L_i: {L_i}")
        # logger.debug(f"L: {L}")
        # logger.debug(f"P: {P}")
        # logger.debug(f"c: {c}")
        # logger.debug(f"l: {l}")
        # logger.debug(f"b: {b}\n")


        L_i = L[(t[j], v[j])]

        c = common_prefix(l + L_i, b)
        l = remainder_suffix(c, l + L_i)
        b = remainder_suffix(c, b)

        for s in T.delta_state_to_chars[t[j + 1]]:
            if s != v[j + 1]:
                x_in = (t[j + 1], s)
                x_out = L[(t[j + 1], s)]
                L[x_in] = l + x_out

        L[(t[j], v[j])] = c

        if t[j + 1] in T.F:
            cacheP = T.psi[t[j + 1]]
            P[t[j + 1]] = l + cacheP
            
    # logger.debug(f"L_i: {L_i}")
    # logger.debug(f"L: {L}")
    # logger.debug(f"P: {P}")
    # logger.debug(f"c: {c}")
    # logger.debug(f"l: {l}")
    # logger.debug(f"b: {b}\n")     
            
    T.lambdaT = L
    T.lambdaT[(t[k], v[k])] = b
    for r in range(k+1, len(v)):
        T.lambdaT[(t[r], v[r])] = ""
        
    T.psi = P
    T.psi[t[-1]] = ""
    
    return T.lambdaT, T.psi  
    

def add_new_entry(T: Transducer, h: Dict[Signature, State], v: str, beta: str, u: str) -> Tuple[Transducer, Dict[Signature, State]]:
    t = state_seq(T.deltaT, T.s, v)
    k = len(t) - 1
    
    T, h = reduce_except(T, u, k, h)
    
    new_states = [State() for _ in range(len(v) - k)]
    t1 = t + new_states
    
    for s in new_states:
        T.Q.add(s)
        T.itr[s] = 0
        
    T.F.add(new_states[-1])
    
    for i in range(k, len(v)):          
        T.deltaT[(t1[i], v[i])] = t1[i+1]
        
        if t1[i] in T.delta_state_to_chars:
            T.delta_state_to_chars[t1[i]] += [v[i]]
        else:
            T.delta_state_to_chars[t1[i]] = [v[i]]
        
        T.itr[t1[i+1]] += 1
        
    # print()
        
    T.delta_state_to_chars[new_states[-1]] = []
    
    T.lambdaT, T.psi = push_output_forward(T, t1, v, beta, k)
    T.iota = common_prefix(T.iota, beta)
    
    # assert set(T.psi.keys()) == T.F, f"\tF: {T.F}\n\tPSI: {T.psi}"
    # assert set(T.delta_state_to_chars.keys()) == T.Q, f"\tQ: {T.Q}\n\tDELTA_CHARS: {T.delta_state_to_chars}"
    # assert set(T.deltaT.keys()) == set(T.lambdaT.keys()), f"\tDELTA: {T.deltaT}\n\tLAMBDA: {T.lambdaT}"
    # assert set(h.keys()) == T.Q, f"Q: {T.Q}\n\th: {h}"
        
    return T, h
    
def timing(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        return (t2 - t1), *res
    return wrapper


@timing
def construct(D: List[Tuple[str, str]], debug: bool = False, path: str = "") -> Tuple[Transducer, Dict[Signature, State]]:
    T, h = construct_from_first_entry(D[0][0], D[0][1])

    for i in range(1, len(D)):
        if i % 10_000 == 0:
           print(f"ENTRY {i}; WORD {D[i][0]}")
            
        T, h = add_new_entry(T, h, D[i][0], D[i][1], D[i - 1][0])
        
        if debug:
            draw_transducer(T, f"{path}_{i}")
            
    T, h = reduce_except(T, D[-1][0], 0, h)

    return T, h


def construct_from_first_entry(a: str, b: str) -> Tuple[Transducer, Dict[Signature, State]]:
    states = [State() for _ in range(len(a) + 1)]

    Q = set(states)
    s = states[0]
    F = set([states[-1]])

    deltaT = {(states[i], a[i]) :states[i + 1] for i in range(len(a))}
    lambdaT = {(states[i], a[i]) : "" for i in range(len(a))}
    iota = b
    psi = {states[-1]: ""}
    itr = {state: 1 for state in states}
    itr[s] = 0

    delta_state_to_chars = {states[i]: [a[i]] for i in range(len(a))}
    delta_state_to_chars[states[-1]] = []

    # assert set(psi.keys()) == F, f"\tF: {F}\n\tPSI: {psi}"
    # assert set(delta_state_to_chars.keys()) == Q, f"\tQ: {Q}\n\tDELTA_CHARS: {delta_state_to_chars}"
    # assert set(deltaT.keys()) == set(lambdaT.keys()), f"\tDELTA: {deltaT}\n\tLAMBDA: {lambdaT}"
    # assert set(h.keys()) == T.Q, f"Q: {T.Q}\n\th: {h}"

    return Transducer(Q, s, F, deltaT, lambdaT, iota, psi, itr, delta_state_to_chars), {}