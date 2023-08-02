import argparse
import os
from typing import List, Tuple

from graphs import draw_transducer
from transducer import construct


def read_file(filename: str) -> List[Tuple[str, str]]:
    print("BEGINNING TO READ THE FILE...")

    D: List[Tuple[str, str]] = []

    with open(filename, "r") as f:
        for l in f.readlines():
            s = l.strip().split("\t", maxsplit=1)
            x = s[0]
            y = s[1]
            D.append((x, y))

    print("READ THE WHOLE FILE")

    return D


parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('-d', '--debug', dest='debug', action='store_true')
parser.add_argument('-t', '--time', dest='measure_time', action='store_true')
parser.add_argument('-s', '--summary', dest='summary', action='store_true')
parser.add_argument('-g', '--graph', dest='draw_graph', action='store_true')
args = parser.parse_args()

filename = args.filename 
# filename = "data/bbb09M"
D = read_file(filename)

time_elapsed, T, h = construct(D, debug=args.debug, path=os.path.basename(filename))
# T, h = construct(D)
if args.measure_time:
    print(f"Time elapsed: {time_elapsed}")
    
if args.summary:
    print("\nSummary vvv")
    print(f'Number of states: {len(T.Q)}')
    print(f'Number of transitions: {len(T.deltaT)}')
    
if args.draw_graph:
    print("\nDrawing transducer...")
    draw_transducer(T, os.path.basename(filename))
    print("Drawn.")
    