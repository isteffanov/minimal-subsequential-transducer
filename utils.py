import time
from typing import List, Tuple


def read_file_dictionary(filename: str) -> List[Tuple[str, str]]:
    print("Reading dictionary file...")

    D: List[Tuple[str, str]] = []

    with open(filename, "r") as f:
        for l in f.readlines():
            s = l.strip().split("\t", maxsplit=1)
            x = s[0]
            y = s[1]
            D.append((x, y))

    print("Dictionary file read.\n")

    return D


def read_file_lexicon(filename: str) -> List[str]:
    print("Reading lexicon file...")

    L: List[Tuple[str, str]] = []

    with open(filename, "r") as f:
        for line in f.readlines():
            L.append(line.strip())

    print("Lexicon file read.\n")

    return L


def timing(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        return (t2 - t1), res
    return wrapper