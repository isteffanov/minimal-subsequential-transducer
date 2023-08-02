# Minimal subsequential transducer

## Abstract

Transducers are abstract mathematical devices for text rewriting. Subsequential transducers are even more powerful machines, capable of containing vast dictionaries
within them for linear text translation.

The idea of this project is to build a minimal subsequential transducer for
a given sorted dictionary in linear time.

## Install

```
pip install -r requirements.txt
python3 main.py data/small -t -s -g

```

## Options

*-g* draws the transducer

*-s* prints summary of the size of the transducer

*-t* shows elapsed time

*-d* debug; draws transducer on every step of the build process

