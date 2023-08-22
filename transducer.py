import os
import logging
from typing import FrozenSet, Set, Dict, Tuple, List

from utils import timing
# from graphs import draw_transducer

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
                 delta_state_to_chars: Dict[State, List[str]],
                 h: Dict[Signature, State],
                 min_except_for: str
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
        
        self.h = h
        
        self.min_except_for = min_except_for
        

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
    outgoing = frozenset((c, T.lambdaT[q, c], T.deltaT[q, c]) for c in T.delta_state_to_chars[q])
    return Signature(is_final, output, outgoing)


def state_seq(deltaT: Dict[Tuple[State, str], State], q: State, w: str) -> List[State]:
    path = [q]
    state = q
    for a in w:
        if (state, a) in deltaT:
            state = deltaT[state, a]
            path.append(state)
        else:
            break
        
    return path


def construct_from_dictionary(D: List[Tuple[str, str]]) -> Transducer:
    
    T = construct_initial(*D[0])
    
    for i, entry in enumerate(D[1:]):
        if i % 10_000 == 0:
           print(f"ENTRY {i}; WORD {D[i][0]}")
           
        T = add_entry_in_order(T, *entry)
    
    T = reduce_except_for_empty_word(T)
    
    return T
    

def construct_initial(input: str, output: str) -> Transducer:
    
    states = [State() for _ in range(len(input) + 1)]
    
    Q = set(states)
    s = states[0]
    F = set([states[-1]])

    deltaT = {(states[i], input[i]) :states[i + 1] for i in range(len(input))}
    lambdaT = {(states[i], input[i]) : "" for i in range(len(input))}
    iota = output
    psi = {states[-1]: ""}
    itr = {state: 1 for state in states}
    itr[s] = 0

    delta_state_to_chars = {states[i]: [input[i]] for i in range(len(input))}
    delta_state_to_chars[states[-1]] = []
    
    min_except_for = input
    
    return Transducer(Q, s, F, deltaT, lambdaT, iota, psi, itr, delta_state_to_chars, {}, min_except_for)


def add_entry_in_order(T: Transducer, input: str, output: str) -> Transducer:
    
    path = state_seq(T.deltaT, T.s, input)
    prefix_length_covered = len(path) - 1
    
    T = reduce_except_for_min(T, prefix_length_covered)
    
    new_states = [State() for _ in range(len(input) - prefix_length_covered)]
    new_path = path + new_states
    
    # add the states of the new path
    for s in new_states:
        T.Q.add(s)
        T.itr[s] = 0
        
    # set finality
    T.F.add(new_path[-1])
    T.psi[new_path[-1]] = ""
    
    # set transitions of the new path
    for i in range(prefix_length_covered, len(input)):          
        T.deltaT[new_path[i], input[i]] = new_path[i+1]
        
        if new_path[i] in T.delta_state_to_chars:
            T.delta_state_to_chars[new_path[i]] += [input[i]]
        else:
            T.delta_state_to_chars[new_path[i]] = [input[i]]
        
        T.itr[new_path[i+1]] += 1
        
    if new_path[-1] not in T.delta_state_to_chars:
        T.delta_state_to_chars[new_path[-1]] = []
    
    # set output of the new path
    T.lambdaT, T.psi = push_output_forward(T, new_path, input, output, prefix_length_covered)
    T.iota = common_prefix(T.iota, output)
    
    T.min_except_for = input
    
    return T
    

# reducing minimal except for word to the length prefix_length_covered
def reduce_except_for_min(T: Transducer, prefix_length_covered: int) -> Transducer:
    path = state_seq(T.deltaT, T.s, T.min_except_for)
    
    # for each state in the path backwards til the goal length
    for i in range(len(path) - 1, prefix_length_covered, -1):
        state = path[i]
        signature = calc_signature(T, state)
        
        if signature not in T.h:
            # if not equivalent state, add it
            T.h[signature] = state
        else:
            # if there is equvalent state, use it, i.e. minimize further
            T.Q.remove(state)
            if state in T.F:
                T.F.remove(state)
                del T.psi[state]
                
            for s in T.delta_state_to_chars[state]:
                T.itr[T.deltaT[state, s]] -= 1
                del T.deltaT[state, s]
                del T.lambdaT[state, s]
                
            del T.delta_state_to_chars[state]
            
            # redirect the transitions to the equivalent state
            T.itr[T.deltaT[path[i-1], T.min_except_for[i-1]]] -= 1
            T.deltaT[path[i-1], T.min_except_for[i-1]] = T.h[signature]
            T.itr[T.h[signature]] += 1
            
    T.min_except_for = T.min_except_for[:prefix_length_covered]
        
    return T


def reduce_except_for_empty_word(T: Transducer) -> Transducer:
    return reduce_except_for_min(T, 0)


def push_output_forward(T: Transducer, new_path: List[State], input: str, output: str, prefix_length_covered: int) -> Tuple[Dict[Tuple[State, str], str], Dict[State, str]]:
    
    c = ""
    l = ""
    b = output
    output_on_current_step = T.iota
    
    c = common_prefix(l + output_on_current_step, b)
    l = remainder_suffix(c, l + output_on_current_step)
    b = remainder_suffix(c, b)
    
    for s in T.delta_state_to_chars[new_path[0]]:
        if s != input[0]:
            x_in = (new_path[0], s)
            x_out = T.lambdaT[x_in]
            T.lambdaT[x_in] = l + x_out
            
    if new_path[0] in T.F:
        cacheP = T.psi[new_path[0]]
        T.psi[new_path[0]] = l + cacheP
        
    
    for j in range(prefix_length_covered):
        output_on_current_step = T.lambdaT[(new_path[j], input[j])]

        c = common_prefix(l + output_on_current_step, b)
        l = remainder_suffix(c, l + output_on_current_step)
        b = remainder_suffix(c, b)

        if j + 1 < len(input):
            for s in T.delta_state_to_chars[new_path[j + 1]]:
                if s != input[j + 1]:
                    x_in = (new_path[j + 1], s)
                    x_out = T.lambdaT[x_in]
                    T.lambdaT[x_in] = l + x_out
                    
            if new_path[j + 1] in T.F:
                cacheP = T.psi[new_path[j + 1]]
                T.psi[new_path[j + 1]] = l + cacheP

        T.lambdaT[(new_path[j], input[j])] = c


        # print(f'c: {c}')
        # print(f'l: {l}')        
        # print(f'b: {b}')
        # print()
        
    # print('=====')
    if len(input) > prefix_length_covered:    
        # set the rest of the output
        T.lambdaT[(new_path[prefix_length_covered], input[prefix_length_covered])] = b
        for r in range(prefix_length_covered+1, len(input)):
            T.lambdaT[new_path[r], input[r]] = ""
            
        T.psi[new_path[-1]] = ""
    else:
        # in case that the added word in a prefix of another
        # set the rest of the output as a state output
        T.psi[new_path[-1]] = b
        for s in T.delta_state_to_chars[new_path[-1]]:
            T.lambdaT[new_path[-1], s] = l + T.lambdaT[new_path[-1], s]
    
    return T.lambdaT, T.psi  


def add_entry_out_of_order(T: Transducer, input: str, output: str) -> Transducer:
    if input == 'da':
        pass
    prefix = traverse_lcp(T, input)    
    T = increase_except(T, prefix)
        
    T = add_entry_in_order(T, input, output)
    T = reduce_except_for_empty_word(T)
    
    return T
    
    
def increase_except(T: Transducer, w: str) -> Transducer:
    state = T.s
    
    k = 0
    while k < len(w) and (state, w[k]) in T.deltaT:
        next_state = T.deltaT[(state, w[k])]
        
        if T.itr[next_state] == 1:
            state = next_state
            del T.h[calc_signature(T, next_state)]
        else:
            T, state = clone(T, state, w[k], next_state)
            
        k += 1
        
    T.min_except_for = w
            
    return T


def clone(T: Transducer, p: State, a: str, q: State) -> Tuple[Transducer, State]:    
    new_q = State()
    
    T.Q.add(new_q)
    if q in T.F:
        T.F.add(new_q)
        T.psi[new_q] = T.psi[q]
        
    T.itr[q] -= 1
    T.deltaT[p, a] = new_q
    T.itr[new_q] = 1
    T.delta_state_to_chars[new_q] = []
    
    for s in T.delta_state_to_chars[q]:
        T.deltaT[new_q, s] = T.deltaT[(q, s)]
        T.lambdaT[new_q, s] = T.lambdaT[(q, s)]
        T.itr[T.deltaT[q, s]] += 1
        T.delta_state_to_chars[new_q].append(s)
        
    
    return T, new_q


def traverse_lcp(T: Transducer, w: str) -> str:
    state = T.s
    prefix = ""
    for a in w:
        if (state, a) in T.deltaT:
            prefix += a
            state = T.deltaT[state, a]
        else:
            break
        
    return prefix

def add_dictionary_out_of_order(T: Transducer, D: List[Tuple[str, str]]) -> Transducer:
    State.state_counter = 1 + max(state.index for state in T.Q)

    for i, entry in enumerate(D):
        if i % 10000 == 0:
            print(f"ENTRY {i}; WORD {entry[0]}")
            
        T = add_entry_out_of_order(T, *entry)
        # draw_transducer(T, f'chupi_{i}')
        
    return T