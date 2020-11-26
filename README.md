# Theta Integration for SpaceRay
Theta batching for SpaceRay package in order to submit Cobalt jobs and run spaces on different GPU nodes. 

### Installation


### In order to use:
- In order to use this package on ThetaGPU, you need two things:
  1) Definition of objective function
  2) `argparse` parsed argument space with the following required components:
    - `--out`: outfile
    - `--json`: json file of hyperparameter bounds
    - `--trials`: number of trials _per space_, not total
    - `--mode`: mode to apply during `tune.run`, defaults to "max" __(optional)__
    - `--metric`: metric used to guide `tune.run` search, defaults to "average_res" __(optional)__
    - `--ray_dir`: directory used to store Ray Tune logging files, defaults to `/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/ray_results` __(optional)__


### Example Usage
 ```
 from argparse import ArgumentParser
 
 ### see ray tune docs for more info on how to define objective function and report metrics to ray tune
 def objective_func(config):
     ### function training and testing using config from tune.run, then report results
     model.train()
     res = model.test()
     res_dict = {}
     res_dict['res'] = res
     tune.report(**res_dict)
     return res
 
 if __name__ == "__main__":
    print("WARNING: default file locations are used to pickle arguments and hyperspaces. "
          "DO NOT RUN MORE THAN ONE EXPERIMENT AT A TIME.")
    print("Creating spaces.")
    parser = ArgumentParser("Run spaceray hyperparameter search on .")
    startTime = time.time()
    ray.init()
    parser.add_argument("-o", "--out")
    parser.add_argument("-m", "--model")
    parser.add_argument("-t", "--trials")
    parser.add_argument("-n", "--nodes", help="Number of GPU nodes to submit on.")
    parser.add_argument("-j", "--json", help="JSON file defining hyperparameter search space")
    arguments = parser.parse_args()
    theta_spaceray.run(objective_func, arguments)
    
    
 ```
