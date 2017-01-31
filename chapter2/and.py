# coding: utf-8


def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    return int(bool(x1 * w1 + x2 * w2 - theta > 0))


def main():
    assert AND(0, 0) == 0
    assert AND(1, 0) == 0
    assert AND(0, 1) == 0
    assert AND(1, 1) == 1

if __name__ == '__main__':
    main()
