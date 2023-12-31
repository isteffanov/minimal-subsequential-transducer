import argparse
import pickle
import os

from utils import read_file_dictionary, timing
from graphs import draw_transducer
from transducer import Transducer


parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('-d', '--debug', dest='debug', action='store_true')
parser.add_argument('-t', '--time', dest='measure_time', action='store_true')
parser.add_argument('-s', '--summary', dest='summary', action='store_true')
parser.add_argument('-g', '--graph', dest='draw_graph', action='store_true')
parser.add_argument('-S', '--serialize', dest='serialize', action='store_true')
args = parser.parse_args()

filename = args.filename 
D = read_file_dictionary(filename)

T = Transducer(D)

# if args.measure_time:
#     print(f"Time elapsed: {time_elapsed}")
    
if args.summary:    
    print("\nSummary vvv")
    print(f'Number of states: {len(T.Q)}')
    print(f'Number of transitions: {len(T.deltaT)}')
    print(f'Number of final states: {len(T.F)}')
    
if args.draw_graph:
    print("\nDrawing transducer...")
    draw_transducer(T, os.path.basename(filename))
    print("Drawn.")
    
if args.serialize:
    with open(f'dumps/{os.path.basename(filename)}.pickle', 'wb+') as file:
        pickle.dump(T, file)
        
    print('\nTransducer serialized.')