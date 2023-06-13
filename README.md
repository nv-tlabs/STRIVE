# Generating Useful Accident-Prone Driving Scenarios via a Learned Traffic Prior (CVPR 2022)
### Davis Rempe, Jonah Philion, Leonidas Guibas, Sanja Fidler, Or Litany
#### [[Project Page](https://nv-tlabs.github.io/STRIVE/)] [[Blog Post](https://developer.nvidia.com/blog/generating-ai-based-accident-scenarios-for-autonomous-vehicles/?ncid=so-link-452504-vt03#cid=av16_so-link_en-us)] [[NVIDIA DRIVE Labs](https://www.youtube.com/watch?v=BXf9xTR6hJo)]

![STRIVE Teaser](strive_teaser.png)

## Environment Setup
> Note: This codebase has been tested primarily on Ubuntu 18.04 with Python 3.6, PyTorch 1.9, and CUDA 11.1.

* Create a virtual environment with `python3 -m venv strive_env` and activate it `source strive_env/bin/activate`. 
* Update pip with `pip install --upgrade pip`
* Install numpy `pip install numpy==1.19.5` first.
* Install remaining dependencies with `pip install -r requirements.txt`. Note this requirements file contains exact versions, but they may need to be changed based on your setup (however, please **ensure the nuScenes devkit is the specified version**, or you may run into issues).

## nuScenes Dataset
Training and testing the traffic model and scenario generation requires downloading the [nuScenes dataset](https://www.nuscenes.org/) (only the metadata is required) and map expansion. After downloading and placing in `data`, the file structure look something like:
```
data/nuscenes
|__ trainval
|______ v1.0-trainval
|___________ *.json
|______ maps
|___________ basemap
|___________ expansion
|___________ prediction
|___________ *.png
```
The code also supports the mini dataset in the exact same structure.

## Additional Downloads
Weights for the pre-trained traffic models can be [downloaded here](https://drive.google.com/drive/folders/1n3j7gT_SA9RoaLosz7z8ngaQhlfV6oRz?usp=sharing) and should be placed in a directory called `model_ckpt` to be used by the config files described below. We provide models trained on _car_ and _trucks_ only (as used in the main paper) and one trained on all categories (discussed in supplementary document).

We also provide the generated scenarios from Section 5.1/5.2 of the paper where both adversarial and solution optimizations succeeded. You can [download them here](https://drive.google.com/drive/folders/10PF_f_5UAWNZlOmHPTi8oiu2qV6hcZCN?usp=sharing) and place them in `data/strive_scenarios`. To use these scenarios in the analyses described below, you will need to update the configs to point to these scenarios accordingly.

Note these models and scenarios were derived from the nuScenes dataset and are thus separately licensed under CC-BY-NC-SA-4.0.

## Training and Testing Traffic Model

### Training
To train the traffic model from scratch on the _car_ and _truck_ categories (same as in the paper), run:
```
python src/train_traffic.py --config ./configs/train_traffic.cfg
```

### Testing
After training the traffic model, we can evaluate performance and visualize samples. To do this for the model introduced in the main paper (trained on _cars_ and _trucks_), run:
```
python src/test_traffic.py --config ./configs/test_traffic.cfg --ckpt path/to/model.pth
```
By default, this runs through a held out nuScenes split, computes various losses and errors, and visualizes random samples. There are options for other evaluations that can be enabled in the config file. A config file for the supplementary version of the model trained on _all_ agent categories is [also provided](./configs/test_traffic_all_cats.cfg).

> Note: by default, this evaluation is performed over trajectories in the full held out nuScenes validation split, not just the official nuScenes prediction challenge trajectories. See the config file to change this.

### Sampling Traffic Model with Refinement Optimization
Future trajectory samples from the traffic model may contain collisions between vehicles or with the non-drivable area, especially in crowded scenes or with long rollout times. To avoid this, we can use a test-time optimization which seeks to eliminate collisions while ensuring traffic is likely under the prior. To generate samples from the model (seeded with nuScenes trajectories) and run the refinement optimization, use:
```
python src/refine_traffic_optim.py --config ./configs/refine_traffic_optim.cfg --ckpt path/to/model.pth
```
By default, this will save the final scenes along with low-quality visualizations of the samples before and after optimization. To produce a high-quality visualization of the optimized samples, another script is provided:
```
python src/viz_scenario_dir.py --scenarios ./out/refine_traffic_optim_out/scenario_results/success --out ./out/refine_traffic_viz_out --viz_video
```

## Adversarial Scenario Generation

### Running Optimization

Adversarial scenarios and their solutions are generated through an optimization procedure that perturbs initial scenes from the nuScenes dataset. To iterate through nuScenes and attempt to generate scenarios and solutions for the _rule-based_ planner, use:
```
python src/adv_scenario_gen.py --config ./configs/adv_gen_rule_based.cfg --ckpt path/to/model.pth
```
Configurations are also provided for the _replay_ planner (see [adv_gen_replay.cfg](./configs/adv_gen_replay.cfg) for car/truck scenarios and [adv_gen_replay_cyclist.cfg](./configs/adv_gen_replay_cyclist.cfg) for scenarios including cyclists and/or pedestrians).

> Note: the default batch size (defined as the approximate number of agents contained across all variable-size scenes being optimized in a batch) in these configs is quite small. It should be increased according to available GPU memory.

By default, optimization saves a visualization of the scenario at different stages of optimization, and a `json` file containing scenario trajectories and other info. Scenarios are saved in folders depending on which optimizations succeeded: both adversarial and solution optimization succeeded (`adv_sol_success`), just solution failed (`sol_failed`), or adversarial failed and thus solution optimization was not performed (`adv_failed`).

### Analyzing Scenarios

#### Qualitative & Quantitative Evaluation

To render a high-quality visualization of the scenarios generated by adversarial optimization, use something like this:
```
 python src/eval_adv_gen.py --out ./out/adv_gen_rule_based_out/eval_results --scenarios ./out/adv_gen_rule_based_out/scenario_results --eval_qual --viz_res adv_sol_success --viz_stage init adv sol --viz_video
```
This [evaluation script](./src/eval_adv_gen.py) is also a good reference for how to load in scenario `json` files.

To quantitatively evaluate and classify the generated scenarios, run:
```
python src/eval_adv_gen.py --out ./out/adv_gen_rule_based_out/eval_results --scenarios ./out/adv_gen_rule_based_out/scenario_results --eval_quant
```
In the output directory, you will see the following outputs for _all scenarios_ where optimization was attempted (`*_all_scenes`), scenarios where adversarial optimization succeeded but solution failed (`*_all_adv`), and scenarios where both adversarial and solution optimization succeeded (`*_adv_sol`):

1. `*_labels.csv` and `scene_distrib.png` - the classification of each scenario based on a given pre-computed clustering (see below).
2. `eval_*.csv` - metrics computed for generated scenario trajectories measuring their plausibility (as in Table 2 of the paper) and the ability to match the planner during optimization (as in Table 1 of the paper). Plausibility metrics are computed for both the colliding agent (`atk`) and all other "controlled" agents (`other`).

#### Planner Evaluation

Lastly, we can roll out a planner (_rule-based_ or _replay_) and evaluate its performance on either generated challenging scenarios or on "regular" scenarios directly from nuScenes. To do this, run:
```
python src/eval_planner.py --config ./configs/eval_planner.cfg
```
By default, this will evaluate the _rule-based_ planner on the scenarios generated from running `adv_scenario_gen.py` above and on their corresponding "regular" scenarios (i.e. the initialization for adversarial optimization). In the config file, make sure `scenario_dir` is set correctly and if you're evaluating the _replay_ planner then change `eval_replay_planner` to true and update the input scenarios path accordingly.

Running `eval_planner` will print out various metrics (as in Table 1 of the paper) corresponding to planner acceleration and collision velocity for the `regular`, adversarial (`adv`), and over all (`total`) scenarios. It additionally outputs a csv file with per-scenario performance.

Note that `eval_planner.py` is a good example of how to use the rule-based planner by itself.

#### Clustering

Generated scenarios can be clustered based on collision properties to analyze the distribution of scenarios. The exact clustering used in the paper is provided in `data/clustering` -- it was performed on a large set (over 400) scenarios generated from various subsets of nuScenes and using many versions of our rule-based planner. This clustering can be used out-of-the-box to classify newly generated scenarios using the `eval_adv_gen` script as detailed previously. However, if you would like to re-run clustering on your own set of scenarios use something like:
```
python src/cluster_scenarios.py --scenario_dirs ./out/adv_gen_rule_based_out/scenario_results/adv_sol_success ./out/adv_gen_rule_based_out/scenario_results/sol_failed --out ./out/rule_based_clustering
```
In this example, all scenarios that caused a collision (even those where a solution was not found) are used for the clustering. This outputs a file called `cluster.pkl` that is fed to the evaluation scripts described above. You can also add the `--viz` flag to visualize the collisions in each cluster. You will also need to define your own `cluster_labels.txt` file in order to run the evaluations above. Please see [our provided clustering](./data/clustering/cluster_labels.txt) for an example; note the labels must be in the corresponding order to the cluster indices.

## Find Out More

For more information and results, check out our [project page](https://nv-tlabs.github.io/STRIVE/). 
STRIVE is also featured on an episode of [NVIDIA Drive Labs](https://www.youtube.com/watch?v=BXf9xTR6hJo)!

## Citation
If you found this code or paper useful, please consider citing:
```
@inproceedings{rempe2022strive,
    author={Rempe, Davis and Philion, Jonah and Guibas, Leonidas J. and Fidler, Sanja and Litany, Or},
    title={Generating Useful Accident-Prone Driving Scenarios via a Learned Traffic Prior},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}  
```

## Questions?
If you run into any problems or have questions, please create an issue or contact Davis via email.
