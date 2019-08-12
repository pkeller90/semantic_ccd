import pandas as pd
import simpletext.alg as alg_st
import keywords_picto.alg as alg_kp
import synth_high.alg as alg_sh
import ast_simple.alg as alg_as
import os
import traceback
from PIL import Image


def visualize(algo, dataset, datasets_path, lazy_mode=False):
    code_ds = pd.read_csv(datasets_path + "/" + dataset + '/funcs.csv')
    if algo == "as":
        vis = alg_as.generate_viz
    elif algo == "sh":
        vis = alg_sh.render
    elif algo == "kp":
        vis = alg_kp.keywords_picto
    elif algo == "st":
        vis = alg_st.text2png
    else:
        print("Unknown algorithm:", algo)

    if lazy_mode:
        out_dir = "%s/images/%s" % (datasets_path, algo)
    else:
        out_dir = "%s/%s/images/%s" % (datasets_path, dataset, algo)

    os.makedirs(out_dir, exist_ok=True)
    for _, r in code_ds.iterrows():
        cid = r["id"]
        out_path = "%s/%s.png" % (out_dir, cid)
        if lazy_mode:
            if os.path.exists(out_path):
                continue
        try:
            img = vis(r["code"])
            img.save(out_path)
        except:
            print("generating image %s-%s: %s failed" % (dataset, algo, cid))
            print(traceback.format_exc())
            print("generating default image as replacement")

            ###########################
            background = (128, 128, 128)
            img = Image.new('RGBA', (448,448), background)
            img.save(out_path)
            ###########################


def try_visualize(algo, dataset, datasets_path, lazy_mode):
    try:
        visualize(algo, dataset, datasets_path, lazy_mode)
    except:
        print("="*40)
        print("Failed to visualize dataset: %s using algorithm %s" % (dataset, algo))
        print(traceback.format_exc())

    print("="*80)

lazy_mode = True #since all current datasets are based on the same data, we can render images per id only once
ds_path = "<path>/datasets" # Path must be set

try_visualize("st", "viscode_t4_limited_art", ds_path, lazy_mode)  #EX1,
try_visualize("sh", "viscode_t4_limited_art", ds_path, lazy_mode)  #EX1,
try_visualize("kp", "viscode_t4_limited_art", ds_path, lazy_mode)  #EX1,
try_visualize("as", "viscode_t4_limited_art", ds_path, lazy_mode)  #EX1,

try_visualize("st", "viscode_t4_limited", ds_path, lazy_mode)  #EX1,
#try_visualize("sh", "viscode_t4_limited", ds_path, lazy_mode)
#try_visualize("kp", "viscode_t4_limited", ds_path, lazy_mode)
try_visualize("as", "viscode_t4_limited", ds_path, lazy_mode)

try_visualize("st", "gen", ds_path, lazy_mode)
try_visualize("sh", "gen", ds_path, lazy_mode)
try_visualize("kp", "gen", ds_path, lazy_mode)
try_visualize("as", "gen", ds_path, lazy_mode)

try_visualize("st", "astnn_t4", ds_path, lazy_mode)
#try_visualize("sh", "astnn_t4", ds_path, lazy_mode)
#try_visualize("kp", "astnn_t4", ds_path, lazy_mode)
try_visualize("as", "astnn_t4", ds_path, lazy_mode)
