import binary_decision.learn as alg_bd
import binary_decision.learn_svm as alg_svm
import func_classification.learn as alg_fc
import astnn.learn as alg_astnn
import os
import time
import traceback
from expreport import report

def try_learn(algo_name, algo, dataset, vis, datasets_path, epochs, lazy_mode):

    print("="*80)
    print("="*80)
    print("Running learning for:")
    print("     Algorithm: %s" % algo_name)
    print("     Dataset:   %s" % dataset)
    print("     Visual.:   %s" % vis)
    print("     CSV-DAT:   %s %s %s" % (algo_name, dataset, vis))
    report.alg = algo_name
    report.dataset = dataset
    report.vis = vis
    print("="*20)
    print()
    try:
        dat_path = "%s/%s/funcs.csv" % (datasets_path, dataset)
        pairs_path = "%s/%s/pairs.csv" % (datasets_path, dataset)
        if lazy_mode:
            img_path = "%s/images/%s" % (datasets_path, vis)
        else:
            img_path = "%s/%s/images/%s" % (datasets_path, dataset, vis)

        algo(dat_path, img_path, pairs_path, epochs)
    except:
        print("="*40)
        print("Failed to learn using alg: %s %s %s" % (algo_name, dataset, vis))
        print(traceback.format_exc())

    print("="*80)
    print("="*80)
    print("="*80)
    print("="*80)
    print()
    print()

lazy_mode = True #since all current datasets are based on the same data, we can render images per id only once
datasets_path = "<path>/datasets" # Path must be set
epochs = 5
algos = {"fc": alg_fc.learn,
         "bdnn": alg_bd.learn,
         "bdsvm": alg_svm.learn,
         "astnn": alg_astnn.learn}
visualizations = ["st", "sh", "kp", "as"]
datasets = ["viscode_t4_limited", "viscode_t4_limited_art", "gen", "astnn_t4"]

#for alg in algos.keys():
#    for vis in visualizations:
#        for dataset in datasets:
#            try_learn(alg, algos[alg], dataset, vis, datasets_path, epochs, lazy_mode)


#for dataset in datasets:
#    try_learn("astnn", alg_astnn.learn, dataset, None, datasets_path, epochs, lazy_mode)

start_time_str = time.strftime("%Y_%m_%d__%H_%M_%S", time.gmtime())

#Experiment 1 - visualization influence
def exp1():
    report.set_save_file("experiment1_%s.csv" % start_time_str)
    for run in range(3):
        report.run = run
        print("-CSV-Run: %s" % run)
        for alg in ["bdsvm"]:
            for vis in ["st", "sh", "kp", "as"]:
                for dataset in ["viscode_t4_limited_art"]:
                    try_learn(alg, algos[alg], dataset, vis, datasets_path, epochs, lazy_mode)
    report.reset()


#Experiment 2 - best classifier algorithm
def exp2():
    report.set_save_file("experiment2_%s.csv" % start_time_str)
    for run in range(3):
        report.run = run
        print("-CSV-Run: %s" % run)
        for alg in ["bdsvm"]: # bdnn bundled with bdsvm now to only generate vectors once...
            for vis in ["as"]:
                for dataset in ["viscode_t4_limited", "viscode_t4_limited_art", "astnn_t4"]:
                    try_learn(alg, algos[alg], dataset, vis, datasets_path, epochs, lazy_mode)
    report.reset()


# Experiment 3 - state of the art
def exp3():
    report.set_save_file("experiment3_%s.csv" % start_time_str)
    for run in range(3):
        report.run = run
        print("--- Run: %s" % run)
        for alg in ["astnn"]:
            for vis in ["as"]:
                for dataset in ["viscode_t4_limited", "viscode_t4_limited_art", "astnn_t4"]:
                    try_learn(alg, algos[alg], dataset, vis, datasets_path, epochs, lazy_mode)
    report.reset()


# Experiment 4 - Generalization abilities
def exp4():
    report.set_save_file("experiment4_%s.csv" % start_time_str)
    for run in range(3):
        report.run = run
        print("--- Run: %s" % run)
        for alg in ["bdsvm", "astnn"]:
            for vis in ["as"]:
                for dataset in ["gen"]:
                    try_learn(alg, algos[alg], dataset, vis, datasets_path, epochs, lazy_mode)
    report.reset()

# Experiment 5 - clone classification evaluation
def exp5():
    report.set_save_file("experiment5_%s.csv" % start_time_str)
    for run in range(3):
        report.run = run
        print("--- Run: %s" % run)
        for alg in ["fc"]:
            for vis in ["st", "sh", "kp", "as"]:
                for dataset in ["viscode_t4_limited_art", "gen"]:
                    try_learn(alg, algos[alg], dataset, vis, datasets_path, epochs, lazy_mode)
    report.reset()


def exptest():
    report.set_save_file("test.csv")
    for run in range(3):
        report.run = run
        print("--- Run: %s" % run)
        for alg in ["fc"]:
            for vis in ["st", "sh", "kp", "as"]:
                for dataset in ["viscode_t4_limited_art", "gen"]:
                    try_learn(alg, algos[alg], dataset, vis, datasets_path, epochs, lazy_mode)
    report.reset()

#exptest()
exp1()
exp2()
exp3()
exp4()
exp5()
