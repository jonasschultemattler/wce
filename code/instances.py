#! /usr/bin/env python3
import os


if __name__ == "__main__":
    file = open("instances-train.txt", "w+")
    for i in range(101, 161, 2):
        instance = "../../wce-students/1-random/r%s.dimacs\n" % str(i).zfill(3)
        file.write(instance)
    for i in range(1, 183, 2):
        instance = "../../wce-students/2-real-world/w%s.dimacs\n" % str(i).zfill(3)
        file.write(instance)
    for i in range(1, 43, 2):
        instance = "../../wce-students/3-actionseq/a%s.dimacs\n" % str(i).zfill(3)
        file.write(instance)
    file.close()
