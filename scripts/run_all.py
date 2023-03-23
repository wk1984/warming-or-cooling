import sys,os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--texture', type=str,
                    help="run files with specific texture/soil composition. one of '02peat', '0peat', '05peat'")

args = parser.parse_args()

tex = str(args.region)
regions=['cd','cw','wd','ww']

for reg in regions:
    runs = ['c', 'i']
    for i in runs:
        rn_cmd = './run.sh &'
        direc = '/home/ats_1_2/modeling/syn_impale/regions/' + reg + '/'tex'/' + i
        os.chdir(direc)
        print(direc)
        os.system(rn_cmd)
