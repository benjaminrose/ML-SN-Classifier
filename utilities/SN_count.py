import os
import sys
import numpy as np
from astropy.table import Table, join, vstack
from astropy.io.ascii import write, read
import traceback
import glob
import argparse
import re

def main(argsdict):

    file_template = argsdict['file_template']
    colname = argsdict['colname']
    Iavalues = argsdict['Iavalues']

    print('Using file template {}\n'.format(file_template))
    files = sorted(glob.glob(file_template+'*'))
    #print(files)
    for filename in files:
        table = read(filename)
        if colname in table.colnames:
            values = list(set(table[colname]))
            print('{}: ({} entries)'.format(filename, len(table)))
            for value in values:
                mask = (table[colname]==value)
                print('   {} entries with value {}'.format(np.count_nonzero(mask), value))

            Iavalues = [Iavalues] if type(Iavalues) is not list else Iavalues
            Iasum = 0
            for Iavalue in Iavalues:
                mask = (table[colname]==Iavalue)
                Iasum += np.count_nonzero(mask)

            print('{} Summary: IA: {} CC: {}\n'.format(filename, Iasum, len(table)-Iasum))
        else:
            print('{} not found in {}\n'.format(colname, filename))

def parse_args(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Count SN types')
    parser.add_argument('file_template', help='Template for filenames to match', default='')
    parser.add_argument('--Iavalues', type=int, help='values for SNIA (101, 1, 0)', default=[0, 1, 101])
    parser.add_argument('--colname', help='Name of column (TYPE, SIM_TYPE_INDEX, SIM_TEMPLATE_INDEX)', default='TYPE')

    args=parser.parse_args()
    argsdict=vars(args)
    print('Running {} with parameters:'.format(sys.argv[0]))
    for arg in argsdict.keys():
        print('{} = {}'.format(arg, argsdict[arg]))

    return argsdict

if __name__=='__main__':
#    try:
        argsdict=parse_args(sys.argv)
        main(argsdict)
#    except:
#        print('')
#        traceback.print_exc()
#
# Example runs
#
#python SN_count.py 'SNIRF_training_sets/*/*/*/output/*/*.FITRES' --colname 'SIM_TEMPLATE_INDEX' --Iavalues 0 > SIM_TEMPLATE_INDEX.stats
#
#python SN_count.py 'SNIRF_training_sets/*/*/*/output/*/*.FITRES' --colname 'SIM_TYPE_INDEX' > SIM_TYPE_INDEX.stats
#
#python SN_count.py 'SNIRF_training_sets/*/*/*/output/*/*.FITRES' --colname 'TYPE' > TYPE.stats
#

