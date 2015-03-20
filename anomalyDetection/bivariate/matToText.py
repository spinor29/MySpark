#!/usr/bin/python

from scipy import io 
import numpy as np
import sys

if __name__ == '__main__':
    infile = sys.argv[1].strip()
    input_data = io.loadmat(infile)
    #print input_data
    X = input_data['X']
    Xval = input_data['Xval']
    yval = input_data['yval']
    with open('X.dat','w') as f:
        for x in X:
            s = ','.join([str(x[0]),str(x[1])])+'\n'
            f.write(s)
    f.close()

    with open('Xval.dat','w') as f:
        for x in Xval:
            s = ','.join([str(x[0]),str(x[1])])+'\n'
            f.write(s)
    f.close()

    with open('yval.dat','w') as f:
        for x in yval:
            s = str(x[0])+'\n'
            f.write(s)
    f.close()
