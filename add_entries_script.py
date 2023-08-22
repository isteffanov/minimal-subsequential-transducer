import argparse
import pickle
import os

from utils import read_file_dictionary, timing
from graphs import draw_transducer
from transducer import add_dictionary_out_of_order


parser = argparse.ArgumentParser()
parser.add_argument('filename')
# parser.add_argument('-d', '--debug', dest='debug', action='store_true')
parser.add_argument('-t', '--time', dest='measure_time', action='store_true')
parser.add_argument('-s', '--summary', dest='summary', action='store_true')
parser.add_argument('-g', '--graph', dest='draw_graph', action='store_true')
parser.add_argument('-S', '--serialize', dest='serialize', action='store_true')
args = parser.parse_args()

filename = args.filename 
# filename = 'data/chupi_tail'
D = read_file_dictionary(filename)

print('Reading serialized transducer...')
with open('transducer.pickle', 'rb') as file:
    T = pickle.load(file)
print('Serialized transducer read.\n')    

@timing
def add_dictionary_out_of_order_timed(T, D):
    return add_dictionary_out_of_order(T, D)
    
time_elapsed, T = add_dictionary_out_of_order_timed(T, D)

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
    
if args.serialize:
    with open('transducer.pickle', 'wb+') as file:
        pickle.dump(T, file)
        
    print('\nTransducer serialized.')