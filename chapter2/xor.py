# coding: utf-8

from .and_node import AND
from .or_node import OR
from .nand_node import NAND


def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))


def main():
    assert XOR(0, 0) == 0
    assert XOR(1, 0) == 0
    assert XOR(0, 1) == 0
    assert XOR(1, 1) == 1

if __name__ == '__main__':
    main()
