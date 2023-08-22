import os
import pygraphviz as pgv

class Transducer:
    pass

def draw_transducer(T: Transducer, filename: str) -> None:
    
    edge_labels = {}
    for f, l in T.deltaT:
        t = T.deltaT[(f, l)]
        o = T.lambdaT[(f, l)]
        if (f, t) in edge_labels:
            edge_labels[(f, t)] += f"/\n{l}:{o}"
        else:
            edge_labels[(f, t)] = f"{l}:{o}"
    
    edges = edge_labels.keys()
    
    G = pgv.AGraph(strict=False, directed=True, comment=T.iota)
    
    for (f, t) in edges:
        G.add_edge(f, t, label=edge_labels[(f, t)])
   
    G.add_node(1, label=T.iota if len(T.iota) > 0 else "''")
    
    
    
    for state in T.F:
        n = G.get_node(state)
        n.attr["peripheries"] = "2"    
        n.attr["label"] += f'\n{T.psi[state]}'
    
    G.layout(prog="dot")
    
    os.makedirs("plots", exist_ok=True)
    G.draw(os.path.join("plots", f"{filename}.png"))    
