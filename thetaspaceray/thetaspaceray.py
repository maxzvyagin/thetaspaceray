import spaceray
import pickle
import os
import stat
from ray import tune
from ray.tune.suggest.skopt import SkOptSearch
from skopt import Optimizer
import dill
dill.settings['recurse'] = True
import cloudpickle
import numpy as np
# from hyper_resilient_experiments import *
from hyper_resilient_experiments.bi_tune import multi_train
from hyper_resilient_experiments import bi_tune


def create_pickles(func, args):
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/thetaspaceray_pickled_func", "wb")
    cloudpickle.dump(func, f)
    f.close()
    space, bounds = spaceray.get_trials(args)
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/thetaspaceray_pickled_spaces", "wb")
    pickle.dump(space, f)
    f.close()
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/thetaspaceray_pickled_bounds", "wb")
    pickle.dump(bounds, f)
    f.close()
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/thetaspaceray_pickled_args", "wb")
    pickle.dump(args, f)
    f.close()
    return space


def chunks(l, n):
    """ Given a list of numbers, return splits based on number of nodes. np array split works because all inputs
    will have length of power of 2"""
    l = np.array(l)
    res = np.split(l, n)
    for i in res:
        yield list(i)


def submit_job(chunk, args):
    command = "qsub -n 1 -A CVD-Mol-AI -t 12:00:00 --attrs pubnet=true "
    chunk_name = str(chunk).replace(" ", "_")
    chunk_name = chunk_name.replace(", ", "")
    script_name = "/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/scripts/script" + args.out + chunk_name + ".sh"
    command = command + script_name
    f = open(script_name, "w")
    f.write("#!/bin/bash\n")
    f.write(
        "singularity shell --nv -B /lus:/lus /lus/theta-fs0/software/thetagpu/nvidia-containers/tensorflow2/tf2_20.10-py3.simg\n")
    # f.write("python /home/mzvyagin/hyper_resilient/theta_batch.py -n " + str(chunk) + "\n")
    python_command = "import thetaspaceray;"
    python_command += "thetaspaceray.run_single(" + str(chunk) + ")"
    f.write("python -c '" + python_command + "'\n")
    f.close()
    st = os.stat(script_name)
    os.chmod(script_name, st.st_mode | stat.S_IEXEC)
    os.system(command)


def run_single(s, mode="max", metric="average_res",
               ray_dir="/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/ray_results"):
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/thetaspaceray_pickled_args", "rb")
    args = pickle.load(f)
    try:
        mode = args.mode
    except AttributeError:
        print("Using default mode.")
    try:
        metric = args.metric
    except AttributeError:
        print("Using default metric average_res.")
    try:
        ray_dir = args.ray_dir
    except:
        print("Using default ray tune results folder.")
    f.close()
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/thetaspaceray_pickled_spaces", "rb")
    hyperspaces = pickle.load(f)
    f.close()
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/thetaspaceray_pickled_bounds", "rb")
    bounds = pickle.load(f)
    f.close()
    # func = bi_tune.multi_train
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/thetaspaceray_pickled_func", "rb")
    func = cloudpickle.load(f)
    f.close()
    for i in s:
        current_space = hyperspaces[i]
        optimizer = Optimizer(current_space)
        search_algo = SkOptSearch(optimizer, list(bounds.keys()), metric=metric, mode=mode)
        try:
            analysis = tune.run(func, search_alg=search_algo, num_samples=int(args.trials),
                                resources_per_trial={'cpu': 25, 'gpu': 1}, local_dir=ray_dir)
            df = analysis.results_df
            df_name = "/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/thetaspaceray/" + args.out + "/"
            df_name += "space_"
            df_name += str(i)
            df_name += ".csv"
            df.to_csv(df_name)
        except:
            print("Couldn't finish space " + str(i) + ".")


def run(args, func):
    """Given objective function and experiment parameters, run spaceray on ThetaGPU"""
    spaces = create_pickles(func, args)
    space_chunks = chunks(list(range(len(spaces))), int(args.nodes))
    # given these space chunks, run in singularity container on GPU node with 12 hr timelimit
    for chunk in space_chunks:
        submit_job(chunk, args)
