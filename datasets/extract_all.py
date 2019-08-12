from paths import *
from extract_vc_dataset import extract_clone_pairs as vc_extract
from extract_vc_dataset import extract_generalization_set as gen_extract
from extract_astnn_dataset import extract as astnn_extract
import os
import traceback
import pandas as pd

def try_extract_vc(dataset_name, t4_only=False, limits=None, artifical_non_clones=True):
    try:
        functionalities = [7, 13, 44]
        pairs, code = vc_extract(functionalities, type4_only=t4_only,
                                 limits=limits, artifical_non_clones=artifical_non_clones)

        folder_path = tool_path + "/datasets/%s" % dataset_name
        os.makedirs(folder_path, exist_ok=True)
        pairs.to_csv(folder_path + "/pairs.csv")
        code.to_csv(folder_path + "/funcs.csv")
    except:
        print("Failed to extract vc dataset %s" % dataset_name)
        print(traceback.format_exc())

    print()
    print("=" * 80)

try_extract_vc("viscode_t4_limited",
               t4_only=True, limits={"0":500, "1":0, "2":0, "3":0, "4":500},
               artifical_non_clones=False)
try_extract_vc("viscode_t4_limited_art",
               t4_only=True, limits={"0":500, "1":0, "2":0, "3":0, "4":500},
               artifical_non_clones=True)


def try_extract_gen(dataset_name, t4_only=False, limits=None, artifical_non_clones=True):
    try:
        train_functionalities = [7, 13, 44]
        test_functionalities = [2, 14, 20, 26]
        pairs, code = gen_extract(train_functionalities, test_functionalities,
                                  t4_only, limits, artifical_non_clones)
        folder_path = tool_path + "/datasets/%s" % dataset_name
        os.makedirs(folder_path, exist_ok=True)
        pairs.to_csv(folder_path + "/pairs.csv")
        code.to_csv(folder_path + "/funcs.csv")
    except:
        print("Failed to extract gen dataset %s" % dataset_name)
        print(traceback.format_exc())

    print()
    print("=" * 80)

try_extract_gen("gen",
                t4_only=True, limits={"0":500, "1":0, "2":0, "3":0, "4":500},
                artifical_non_clones=True)


def try_extract_astnn(dataset_name, t4_only=True):
    try:
        pairs, code = astnn_extract(t4_only)

        folder_path = tool_path + "/datasets/%s" % dataset_name
        os.makedirs(folder_path, exist_ok=True)
        pairs.to_csv(folder_path + "/pairs.csv")
        code.to_csv(folder_path + "/funcs.csv")
    except:
        print("Failed to extract astnn dataset %s" % dataset_name)
        print(traceback.format_exc())

    print()
    print("=" * 80)

#try_extract_astnn("astnn_t4", True)
