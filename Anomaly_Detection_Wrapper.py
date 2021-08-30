import os
import argparse
import subprocess

def main(args):
    casePath = args.casePath

def make_parser():
    parser = argparse.ArgumentParser(description = 'Wrapping Anomaly_Detction_ALgorithm')
    parser.add_argument('-casePath', type=str, required=True, help = 'Path to directory containing the nifti, the labeled graph, the segments array and the labeled Excel')
    return parser

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
  
# parser = argparse.ArgumentParser()
# parser.add_argument('-casePath', '--casePath', type=str, required=True, help ='Path to directory containing the nifti, the labeled graph, the segments array and the labeled Excel')
# args = parser.parse_args()
main(args)
casePath = args.casePath
d = vars(args)

slicerPath = '/Applications/Slicer.app/Contents/MacOS/Slicer'
#slicerPath = '/Applications/Slicer.app'
caseDir = os.path.abspath(os.path.dirname(casePath))
Anomaly_Detection_CodePath = '/Users/projectephantom/Desktop/Marc/Anomaly_Detection.py'

os.system(f'{slicerPath} --python-script {Anomaly_Detection_CodePath} -casePath {casePath} --exit-after-startup')

# --no-main-window 
