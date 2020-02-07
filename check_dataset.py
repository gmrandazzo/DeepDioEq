#!/usr/bin/env python
import sys

def print_smallnumbers(v):
    if len(v[1])<4 and len(v[2])<4 and len(v[3])<4:
        print("%s,%s,%s,%s" % (v[0], v[1], v[2], v[3]))


def cube(x_):
    x = int(x_)
    return x*x*x

def check(fcsv):
    f = open(fcsv, "r")
    for line in f:
        if "n" in line:
            continue
        else:
            v = str.split(line.strip(), ",")
            n = cube(v[1])+cube(v[2])+cube(v[3])
            if int(n) == int(v[0]):
                # Only to filter the dataset...
                # print_smallnumbers(v)
                continue
            else:
                print("problem with line %s" % (line))
    f.close()

def main():
    if len(sys.argv) != 2:
        print("\nUsage: %s [dataset.csv file]" % (sys.argv[0]))
    else:
        check(sys.argv[1])
    return 0

if __name__ in "__main__":
    main()

