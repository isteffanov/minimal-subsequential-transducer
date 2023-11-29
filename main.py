import argparse
import pickle
import os

from utils import read_file_dictionary
from transducer import Transducer


parser = argparse.ArgumentParser()
parser.add_argument('filename')
# parser.add_argument('-a', '--add', dest='file_to_add', type=str, required=False)
parser.add_argument('-r', '--remove', dest='file_to_remove', type=str, required=True)
parser.add_argument('-t', '--time', dest='measure_time', action='store_true')
parser.add_argument('-s', '--summary', dest='summary', action='store_true')
parser.add_argument('-g', '--graph', dest='draw_graph', action='store_true')
parser.add_argument('-S', '--serialize', dest='serialize', action='store_true')
parser.add_argument('-c', '--check', dest='check', action='store_true')

args = parser.parse_args()

filename = args.filename 
file_to_remove = args.file_to_remove


D = read_file_dictionary(filename)
DR = read_file_dictionary(file_to_remove)
L = list(map(lambda x: x[0], DR))


if os.path.exists(f'dumps/{filename}_rem_{file_to_remove}.pickle'):
    with open(f'dumps/{filename}_rem_{file_to_remove}.pickle', 'rb') as file:
        T = pickle.load(file)
        
elif os.path.exists(f'dumps/{filename}.pickle'):
    with open(f'dumps/{filename}.pickle', 'rb') as file:
        T = pickle.load(file)
        
    T.remove_lexicon_out_of_order(L)
        
else:
    T = Transducer(D)

    if args.serialize:
        with open(f'dumps/{filename}.pickle', 'wb') as file:
            pickle.dump(T, file)
        
    T.remove_lexicon_out_of_order(L)
    
    if args.serialize:
        with open(f'dumps/{filename}_rem_{file_to_remove}.pickle', 'wb') as file:
            pickle.dump(T, file)
        
if args.summary:
    print("Summary (GOT) vvv")
    print(f'Number of states: {len(T.Q)}')
    print(f'Number of transitions: {len(T.deltaT)}')
    print(f'Number of final states: {len(T.F)}')
    print()
    
if args.check:

    DA = sorted(list(set(D) - set(DR)))
    TA = Transducer(DA)

    if args.summary:
        print("SHOULD GET vvv")
        print(f'Number of states: {len(TA.Q)}')
        print(f'Number of transitions: {len(TA.deltaT)}')
        print(f'Number of final states: {len(TA.F)}')
        
    with open(f'dumps/sure.pickle', 'wb') as file:
            pickle.dump(TA, file)
   