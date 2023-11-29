import os
import logging
from typing import FrozenSet, Set, Dict, Tuple, List

from utils import timing
# from graphs import draw_transducer

logger = logging.getLogger(__name__)
logger.setLevel(3)

SIGMA = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
         'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'self', 'U', 'V', 'W', 'X', 'Y',
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
                 revDelta: Dict[State, Set[Tuple[State, str]]],
                 delta_state_to_chars: Dict[State, Set[str]],
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
        self.revDelta = revDelta
        self.delta_state_to_chars = delta_state_to_chars
        
        self.h = h
        
        self.min_except_for = min_except_for
        
    def __init__(self, D: List[Tuple[str, str]]) -> "Transducer":
    
        self.construct_initial(*D[0])
        
        for i, entry in enumerate(D[1:]):
            if i % 10_000 == 0:
                print(f"ENTRY {i}; WORD {D[i][0]}")
            
            self.add_entry_in_order(*entry)

        
        self.reduce_except_for_empty_word()
        
    def itr(self, state: State) -> int:
        return len(self.revDelta[state])
        
            
    def construct_initial(self, input: str, output: str) -> "Transducer":
    
        states = [State() for _ in range(len(input) + 1)]
        
        self.Q = set(states)
        self.s = states[0]
        self.F = set([states[-1]])
        
        self.deltaT = {(states[i], input[i]) :states[i + 1] for i in range(len(input))}
        self.lambdaT = {(states[i], input[i]) : "" for i in range(len(input))}
        self.iota = output
        self.psi = {states[-1]: ""}
        self.revDelta = {states[i + 1]: {(states[i], input[i])} for i in range(len(input))}
        self.revDelta[self.s] = set()

        self.delta_state_to_chars = {states[i]: {input[i]} for i in range(len(input))}
        self.delta_state_to_chars[states[-1]] = set()
        self.h = {}
        self.min_except_for = input
        
        # return Transducer(Q, s, F, deltaT, lambdaT, iota, psi, itr, revDelta,delta_state_to_chars, {}, min_except_for)

    def calc_signature(self, q: State) -> Signature:
        is_final = q in self.F
        output = self.psi[q] if is_final else ""
        outgoing = frozenset((c, self.lambdaT[q, c], self.deltaT[q, c]) for c in self.delta_state_to_chars[q])
        return Signature(is_final, output, outgoing)
    
    def add_entry_in_order(self: "Transducer", input: str, output: str):
        
        path = state_seq(self.deltaT, self.s, input)
        prefix_length_covered = len(path) - 1
        
        self.reduce_except_for_min(prefix_length_covered)
        
        new_states = [State() for _ in range(len(input) - prefix_length_covered)]
        new_path = path + new_states
        
        # add the states of the new path
        for s in new_states:
            self.Q.add(s)
            self.revDelta[s] = set()
            
        # set finality
        self.F.add(new_path[-1])
        self.psi[new_path[-1]] = ""
        
        # set transitions of the new path
        for i in range(prefix_length_covered, len(input)):          
            self.deltaT[new_path[i], input[i]] = new_path[i+1]
            
            if new_path[i] in self.delta_state_to_chars:
                self.delta_state_to_chars[new_path[i]].add(input[i])
            else:
                self.delta_state_to_chars[new_path[i]] = {input[i]}
            
            self.revDelta[new_path[i+1]].add((new_path[i], input[i]))
            
        if new_path[-1] not in self.delta_state_to_chars:
            self.delta_state_to_chars[new_path[-1]] = set()
        
        # set output of the new path
        self.push_output_forward(new_path, input, output, prefix_length_covered)
        self.iota = common_prefix(self.iota, output)
        
        self.min_except_for = input
        
        

    # reducing minimal except for word to the length prefix_length_covered
    def reduce_except_for_min(self: "Transducer", prefix_length_covered: int) -> "Transducer":
        path = state_seq(self.deltaT, self.s, self.min_except_for)
        
        # for each state in the path backwards til the goal length
        for i in range(len(path) - 1, prefix_length_covered, -1):
            state = path[i]
            signature = self.calc_signature(state)
            
            if signature not in self.h:
                # if not equivalent state, add it
                self.h[signature] = state
            else:
                # if there is equvalent state, use it, i.e. minimize further
            
                # self.delete_state_without_incomming_transitions(state) 
                output = self.lambdaT[path[i-1], self.min_except_for[i-1]]
                
                self.fully_delete_state(state)
                    
                self.deltaT[path[i-1], self.min_except_for[i-1]] = self.h[signature]
                self.lambdaT[path[i-1], self.min_except_for[i-1]] = output
                
                self.delta_state_to_chars[path[i-1]].add(self.min_except_for[i-1])
                self.revDelta[self.h[signature]].add((path[i-1], self.min_except_for[i-1]))                       
                
        self.min_except_for = self.min_except_for[:prefix_length_covered]
            
        return self


    def reduce_except_for_empty_word(self: "Transducer") -> "Transducer":
        return self.reduce_except_for_min(0)


    def push_output_forward(self: "Transducer", new_path: List[State], input: str, output: str, prefix_length_covered: int) -> Tuple[Dict[Tuple[State, str], str], Dict[State, str]]:
        
        c = ""
        l = ""
        b = output
        output_on_current_step = self.iota
        
        c = common_prefix(l + output_on_current_step, b)
        l = remainder_suffix(c, l + output_on_current_step)
        b = remainder_suffix(c, b)
        
        for s in self.delta_state_to_chars[new_path[0]]:
            if s != input[0]:
                x_in = (new_path[0], s)
                x_out = self.lambdaT[x_in]
                self.lambdaT[x_in] = l + x_out
                
        if new_path[0] in self.F:
            cacheP = self.psi[new_path[0]]
            self.psi[new_path[0]] = l + cacheP
            
        
        for j in range(prefix_length_covered):
            output_on_current_step = self.lambdaT[(new_path[j], input[j])]

            c = common_prefix(l + output_on_current_step, b)
            l = remainder_suffix(c, l + output_on_current_step)
            b = remainder_suffix(c, b)

            if j + 1 < len(input):
                for s in self.delta_state_to_chars[new_path[j + 1]]:
                    if s != input[j + 1]:
                        x_in = (new_path[j + 1], s)
                        x_out = self.lambdaT[x_in]
                        self.lambdaT[x_in] = l + x_out
                        
                if new_path[j + 1] in self.F:
                    cacheP = self.psi[new_path[j + 1]]
                    self.psi[new_path[j + 1]] = l + cacheP

            self.lambdaT[(new_path[j], input[j])] = c

        if len(input) > prefix_length_covered:    
            # set the rest of the output
            self.lambdaT[(new_path[prefix_length_covered], input[prefix_length_covered])] = b
            for r in range(prefix_length_covered+1, len(input)):
                self.lambdaT[new_path[r], input[r]] = ""
                
            self.psi[new_path[-1]] = ""
        else:
            # in case that the added word in a prefix of another
            # set the rest of the output as a state output
            self.psi[new_path[-1]] = b
            for s in self.delta_state_to_chars[new_path[-1]]:
                self.lambdaT[new_path[-1], s] = l + self.lambdaT[new_path[-1], s]
        

    def add_entry_out_of_order(self: "Transducer", input: str, output: str) -> "Transducer":
        prefix = self.traverse_lcp(input)    
        self.increase_except(prefix)
            
        self.add_entry_in_order(input, output)
        self.reduce_except_for_empty_word()
                
        
    def increase_except(self: "Transducer", w: str) -> "Transducer":
        state = self.s
        
        k = 0
        while k < len(w) and (state, w[k]) in self.deltaT:
            next_state = self.deltaT[(state, w[k])] 
                
            if not self.is_state_convergent(next_state):
                state = next_state
                signature = self.calc_signature(next_state)
                
                del self.h[signature]
            else:
                state = self.clone(state, w[k], next_state)
                
            k += 1
            
        self.min_except_for = w


    def is_state_convergent(self: "Transducer", state: State) -> bool:
        return self.itr(state) > 1


    def clone(self: "Transducer", p: State, a: str, q: State) -> State:    
        new_q = State()
        
        self.Q.add(new_q)
        if q in self.F:
            self.F.add(new_q)
            self.psi[new_q] = self.psi[q]
            
        self.revDelta[q].remove((p, a))
        self.deltaT[p, a] = new_q
        self.revDelta[new_q] = {(p, a)}
        self.delta_state_to_chars[new_q] = set()
        
        for s in self.delta_state_to_chars[q]:
            self.deltaT[new_q, s] = self.deltaT[(q, s)]
            self.lambdaT[new_q, s] = self.lambdaT[(q, s)]
            self.revDelta[self.deltaT[(q, s)]].add((new_q, s))
            self.delta_state_to_chars[new_q].add(s)
        
        return new_q


    def traverse_lcp(self: "Transducer", w: str) -> str:
        state = self.s
        prefix = ""
        for a in w:
            if (state, a) in self.deltaT:
                prefix += a
                state = self.deltaT[state, a]
            else:
                break
            
        return prefix

    def add_dictionary_out_of_order(self: "Transducer", D: List[Tuple[str, str]]) -> "Transducer":
        State.state_counter = 1 + max(state.index for state in self.Q)

        for i, entry in enumerate(D):
            if i % 10000 == 0:
                print(f"ENTRY {i}; WORD {entry[0]}")
                
            # file.write(f'WORD: {entry[0]}; STATES: {len(self.Q)}; TRANS: {len(self.deltaT)}\n')
            self.add_entry_out_of_order(*entry)  
                     


    def remove_lexicon_out_of_order(self: "Transducer", L: List[str]) -> "Transducer":
        State.state_counter = 1 + max(state.index for state in self.Q)

        for i, entry in enumerate(L):
            if i % 10000 == 0:
                print(f"ENTRY {i}; WORD {entry}")
                
            self.remove_entry_out_of_order(entry)
                        

    def remove_entry_out_of_order(self: "Transducer", v: str) -> "Transducer":
        
        self.increase_except(v)
        self.remove_entry_and_pull_output(v)
        self.reduce_except_for_empty_word()
        

    def remove_entry_and_pull_output(self: "Transducer", v: str) -> "Transducer":
        
        t = state_seq(self.deltaT, self.s, v)
        
        self.F.remove(t[-1])
        
        i = 1
        p = t[-i]
        
        non_divergent_states = []
        
        if self.is_state_dead_end(p):
            while not self.is_state_divergent(p):
                non_divergent_states.append(p)
                self.min_except_for = self.min_except_for[:-1]
                i += 1
                p = t[-i]
            
           
            for state in non_divergent_states:
                self.fully_delete_state(state)
           
        self.pull_output_backward()
        
        del self.psi[t[-1]]
        

    def pull_output_backward(self: "Transducer") -> "Transducer":
        path = state_seq(self.deltaT, self.s, self.min_except_for)
        i = 1
        state = path[-i]  
        carry = self.find_pullable_output(state)
       
        while len(path) > 1:
            
            prev_div_state = self.find_prev_divergent_state(path[:-1])
            letter = self.find_letter(prev_div_state, path)
            
            self.lambdaT[prev_div_state, letter] += carry

            carry = self.find_pullable_output(prev_div_state)
            if carry == "":
                break
            
            path = [path[i] for i in range(len(path)) if i < path.index(prev_div_state)]
            
        self.iota += carry
                
    def find_letter(self, state, path):
        letter = ""
        for l in self.delta_state_to_chars[state]:
            if self.deltaT[state, l] in path:
                letter = l
                break
            
        return letter
                

    def find_prev_divergent_state(self: "Transducer", sequence: List[State]) -> State:
        i = 1
        p = sequence[-i]
        
        while not self.is_state_divergent(p):
            i += 1
            p = sequence[-i]
            
            
        return p
                
    def find_pullable_output(self: "Transducer", state: State) -> str: 
        
        state_outputs = [self.lambdaT[state, letter] for letter in self.delta_state_to_chars[state]]
        if state in self.F:
            state_outputs.append(self.psi[state])
            
        lcp_state_outputs = common_prefix_many(*state_outputs)
        
        if state in self.F:
            self.psi[state] = remainder_suffix(lcp_state_outputs, self.psi[state])
            
        for letter in self.delta_state_to_chars[state]:
            self.lambdaT[state, letter] = remainder_suffix(lcp_state_outputs, self.lambdaT[state, letter])
            
        return lcp_state_outputs



    def is_state_dead_end(self: "Transducer", state: State) -> bool:
        return len(self.delta_state_to_chars[state]) == 0


    def is_state_divergent(self: "Transducer", state: State) -> bool:
             return state in self.F or len(self.delta_state_to_chars[state]) > 1


    def fully_delete_state(self: "Transducer", state: State) -> "Transducer":
        if state == self.s:
            raise Exception("Cannot delete start state")
        
        for ch in self.delta_state_to_chars[state]:
            self.revDelta[self.deltaT[state, ch]].remove((state, ch))
            del self.deltaT[state, ch]
            del self.lambdaT[state, ch]
            
        del self.delta_state_to_chars[state]
        
        for q, ch in self.revDelta[state]:
            del self.deltaT[q, ch]
            del self.lambdaT[q, ch]    
            self.delta_state_to_chars[q].remove(ch)

        del self.revDelta[state]

        # self.delete_state_without_incomming_transitions(state)
        
        self.Q.remove(state)
        if state in self.F:
            self.F.remove(state)
            del self.psi[state]
        

def common_prefix(s1: str, s2: str) -> str:
    n = min(len(s1), len(s2))
    for i in range(n):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[:n]


def common_prefix_many(*args: List[str]) -> str:
    return os.path.commonprefix(args)


def remainder_suffix(w: str, s: str) -> str:
    return s[len(w):]


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

    