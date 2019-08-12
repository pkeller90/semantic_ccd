import sys
sys.path.append("..")

import subprocess
import os
from astnn.convert_to_astnn_format import convert
import pandas as pd
import expreport as report

astnn_path = "<astnn_path>/clone" #Path must be set

def exec_command(command):
    subprocess.run(command, shell=True)

    #pipelineproc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #while True:
    #    line = proc.stdout.readline()
    #    if not line:
    #        break
    #    print(line)            

#imagepath and epochs not used
def learn(data_path, image_path, pairs_path=None, epochs=5):
    codes = pd.read_csv(data_path)
    pairs = pd.read_csv(pairs_path)

    n_pairs, n_funcs = convert(pairs, codes)



    outpath = "%s/data/java" % (astnn_path)


    delcom = "echo 'removing old data'; cd '%s'; rm -rf dev; rm -rf test; rm -rf train; rm ast.pkl; echo 'done removing'" % outpath
    exec_command(delcom)
    n_pairs.to_pickle(outpath + "/bcb_pair_ids.pkl")
    n_funcs.to_csv(outpath + "/bcb_funcs_all.tsv", sep="\t", header=None)

    with open("%s/experiment_csv_path.txt" % (astnn_path), "w+") as epout:
        full_report_path = report.report.get_full_outpath()
        epout.write(full_report_path)

    with open("%s/dataset_name.txt" % (astnn_path), "w+") as epout:
        ds_name = report.report.dataset
        epout.write(ds_name)

    pipelinecom = "echo 'start pipeline'; cd %s; python3 pipeline.py --lang java; echo 'finished pipeline'" % astnn_path
    exec_command(pipelinecom)
    print()

    traincom = "echo 'start training'; cd %s; python3 train.py --lang java; echo 'finished training'" % astnn_path
    exec_command(traincom)
