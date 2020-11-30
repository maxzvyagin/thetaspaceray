import spaceray
import pickle
import os
import stat
from ray import tune
from ray.tune.suggest.skopt import SkOptSearch
from skopt import Optimizer
import dill

def create_pickles(func, args):
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/thetaspaceray_pickled_func", "wb")
    dill.dump(func, f)
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
    """ Given a list l, split into equal chunks of length n"""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def submit_job(chunk, args):
    command = "qsub -n 1 -A CVD-Mol-AI -t 12:00:00 --attrs pubnet=true "
    chunk_name = str(chunk).replace(" ", "_")
    chunk_name = chunk_name.replace(", ", "")
    script_name = "/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/scripts/script" + args.out + chunk_name + ".sh"
    command = command + script_name
    f = open(script_name, "w")
    f.write(
        "singularity shell --nv -B /lus:/lus /lus/theta-fs0/software/thetagpu/nvidia-containers/tensorflow2/tf2_20.10-py3.simg\n")
    # f.write("python /home/mzvyagin/hyper_resilient/theta_batch.py -n " + str(chunk) + "\n")
    python_command = "import thetaspaceray;"
    python_command += "thetaspaceray.run_single("+str(chunk)+") \n"
    f.write("python -c " + python_command)
    f.close()
    st = os.stat(script_name)
    os.chmod(script_name, st.st_mode | stat.S_IEXEC)
    os.system(command)


def run_single(s, mode="max", metric="average_res",
               ray_dir="/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/ray_results"):
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/thetaspaceray_pickled_args", "rb")
    args = pickle.load(f)
    if args.mode:
        mode = args.mode
    if args.metric:
        metric = args.metric
    if args.ray_dir:
        ray_dir = args.ray_dir
    f.close()
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/thetaspaceray_pickled_spaces", "rb")
    hyperspaces = pickle.load(f)
    f.close()
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/thetaspaceray_pickled_bounds", "rb")
    bounds = pickle.load(f)
    f.close()
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/thetaspaceray_pickled_func", "rb")
    func = dill.load(f)
    f.close()
    for i in s:
        current_space = hyperspaces[i]
        optimizer = Optimizer(current_space)
        search_algo = SkOptSearch(optimizer, list(bounds.keys()), metric=metric, mode=mode)
        try:
            analysis = tune.run(func, search_alg=search_algo, num_samples=args.trials,
                                resources_per_trial={'cpu': 25, 'gpu': 1},
                                local_dir=ray_dir)
            df = analysis.results_df
            df_name = "/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/thetaspaceray/" + args.out + "/"
            df_name += "space_"
            df_name += str(i)
            df_name += ".csv"
            df.to_csv(df_name)
            print("Finished space " + str(i))
    print("Finished all spaces. Files written to /lus/theta-fs0/projects/CVD-Mol-AI/"
          "mzvyagin/thetaspaceray/" + args.out)


def run(args, func):
    """Given objective function and experiment parameters, run spaceray on ThetaGPU"""
    spaces = create_pickles(func, args)
    space_chunks = chunks(list(range(len(spaces))), int(args.nodes))
    # given these space chunks, run in singularity container on GPU node with 12 hr timelimit
    for chunk in space_chunks:
        submit_job(chunk, args)