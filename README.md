# Dream

## PerAct

### Building the container
To build the Apptainer SIF container, use the DEF file defined at `build_peract_jammy.def`. To run it, execute `sbatch snellius_env/build_peract_jammy.job`. This will generate the SIF container at the project root named `peract_jammy.sif`.

### Data preparation
To get the data you have to first fetch the RLBench data and then generate the DreMa data. The second step can be done following COPPELIA.md which can be found at DreMa's official repository. After loading both data, merge them in order to train a model that includes both types of data. This should happen at the level of the folder named 'episodes'.

### Training
To train the model, run `sbatch snellius_env/train_everything.job`, but first make sure to alter the data path inside the same file. The path should go to a depth just before the task name (in this case slide_block_to_color_target). For example, if the path is `/home/user/scratch/slide_block_to_color_target/all_variations/episodes...` then the given path should be `/home/user/scratch/`.

### Evaluation
To evaluate the model, run `sbatch snellius_env/validate_peract.job`, but first make sure to alter the data path inside the same file. The path should go to a depth just before the task name (in this case slide_block_to_color_target). For example, if the path is `/home/user/scratch/slide_block_to_color_target/all_variations/episodes...` then the given path should be `/home/user/scratch/`. In addition, make sure to give the correct path for the trained weights and create the corresponding folders in which the evaluation will store the results. Details are given at the README.md located at PerAct's official repository. Note that evaluation scripts are not running correctly for PerAct as it is an open issue both for PerAct and this repository.


## 3D Diffuser Actor

### Building the container

To build the Apptainer SIF container using the DEF file defined at `diffuser_actor_jammy.def`, run `sbatch snellius_env/build_diffuser_actor_jammy.job`. This will generate the SIF container at the project root named `diffuser_actor_jammy.sif`.

### Data preparation
Following 3D Diffuser Actor's [data preparation readme](https://github.com/nickgkan/3d_diffuser_actor/blob/e3efaa9a5f7f6fe40de5511ca645295f7b0230b9/docs/DATA_PREPARATION_RLBENCH.md), the data preparation steps are: 1. Rerender, 2. rearrange, 3. package. The `sbatch snellius_env/3d_diff_data_repackaging.job` does all of these steps (this does not work due to the defined container not supporting CoppeliaSim on Snellius's headless environemt at runtime) and the `sbatch snellius_env/3d_diff_data_repackaging_only.job` does step 2 and 3 without requiring CoppeliaSim.

`sbatch snellius_env/3d_diff_test_data_rearrange.job` will rearrange only the RLBench test data on the scratch-shared directory. Export the env variable `HOST_INPUT_FOR_REPACKAGE_ROOT` with where the original RLBench test data is placed, currently it points to `/scratch-shared/tmp.lUdVGE8VOd/3d_diff_packaged/test`, and running the job will rearragne the data for running 3D Diffuser Actor testing script.

#### Creating the mixed dataset
For the RLBench + DreaMa dataset (referred to as “mixed”), the data must first be unzipped, merged into a single directory, and then uniformly renamed. DreaMa episodes follow the `episodeXXXX` format (e.g., `episode0023`), while RLBench uses `episodeXX` (e.g., `episode23`), which breaks the 3D Diffuser Actor’s repackaging scripts. To avoid this, all episodes should be renamed to the `episodeXX` format after merging. Once this is done, the `3d_diff_data_repackaging_only.job` script can be used to prepare the data for use with 3D Diffuser Actor.

### Training and Evaluating

To run the training job on the `slide_block_to_color_target` task, export the env variable `HOST_PERACT_PACKAGED_DATA_ROOT` with the location of the training data. For example, for the origianl RLBench data we point to `"/scratch-shared/tmp.lUdVGE8VOd/Peract_onetask_repackaged"` and for the mixed data, we use `"/scratch-shared/tmp.lUdVGE8VOd/mixed_onetask_repackaged"` in the job file `snellius_env/3d_diff_training.job`. This binds the scratch-shared directory on Snellius where we have placed the task data repacked for 3D Diffuser Actor, which we repackaged following the [data preparation readme](https://github.com/nickgkan/3d_diffuser_actor/blob/e3efaa9a5f7f6fe40de5511ca645295f7b0230b9/docs/DATA_PREPARATION_RLBENCH.md) provided by the project. We do not upscale the image for a fair comparison with PerAct.

Run the training job with `sbatch snellius_env/3d_diff_training.job`.

To evaluate the trained policy, edit `HOST_CHECKPOINT_FILE_PATH` to point to the saved checkpoint file and run `sbatch snellius_env/3d_diff_eval.job`. This currently does not work as the container does not support CoppeliaSim on Snellius.


## Test-Time Adaptation (TTA)

This is in an incomplete stage and is reported only for documentation purposes. The corresponding files can be found under the 'TTA' folder.

