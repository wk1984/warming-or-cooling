import sys,os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--texture', type=str,
                    help="run files with specific texture/soil composition. one of '02peat', '0peat', '05peat'")

args = parser.parse_args()
basepath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

tex = str(args.texture)
regions=['cd','cw','wd','ww']

for reg in regions:
    runs = ['c', 'i']
    for i in runs:
        rn_cmd = './run.sh &'
        direc = basepath + '/regions/' + reg + '/'tex'/' + i
        os.chdir(direc)
        print(direc)
        os.system(rn_cmd)
