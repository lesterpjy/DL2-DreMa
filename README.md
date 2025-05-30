# Dream



## 3D Diffuser Actor

### Building the container

To build the Apptainer SIF container using the DEF file defined at `diffuser_actor_jammy.def`, run `sbatch snellius_env/build_diffuser_actor_jammy.job`. This will generate the SIF container at the project root named `diffuser_actor_jammy.sif`.

### Data preparation
Following 3D Diffuser Actor's [data preparation readme](https://github.com/nickgkan/3d_diffuser_actor/blob/e3efaa9a5f7f6fe40de5511ca645295f7b0230b9/docs/DATA_PREPARATION_RLBENCH.md), the data preparation steps are: 1. Rerender, 2. rearrange, 3. package. The `sbatch snellius_env/3d_diff_data_repackaging.job` does all of these steps (this does not work due to the defined container not supporting CoppeliaSim on Snellius's headless environemt at runtime) and the `sbatch snellius_env/3d_diff_data_repackaging_only.job` does step 2 and 3 without requiring CoppeliaSim.

`sbatch snellius_env/3d_diff_test_data_rearrange.job` will rearrange only the RLBench test data on the scratch-shared directory. Export the env variable `HOST_INPUT_FOR_REPACKAGE_ROOT` with where the original RLBench test data is placed, currently it points to `/scratch-shared/tmp.lUdVGE8VOd/3d_diff_packaged/test`, and running the job will rearragne the data for running 3D Diffuser Actor testing script.

### Training and Evaluating

To run the training job for the original RLBench data on the `slide_block_to_color_target` task, uncomment `HOST_PERACT_PACKAGED_DATA_ROOT="/scratch-shared/tmp.lUdVGE8VOd/Peract_onetask_repackaged"` and comment `HOST_PERACT_PACKAGED_DATA_ROOT="/scratch-shared/tmp.lUdVGE8VOd/mixed_onetask_repackaged"` in the job file `snellius_env/3d_diff_training.job`. This binds the scratch-shared directory on Snellius where we have placed the original task data repacked for 3D Diffuser Actor, which we repackaged following the [data preparation readme](https://github.com/nickgkan/3d_diffuser_actor/blob/e3efaa9a5f7f6fe40de5511ca645295f7b0230b9/docs/DATA_PREPARATION_RLBENCH.md) provided by the project. We do not upscale the image for fair comparison.

Similarly we repakcaged DreMa's `slide_block_to_color_target` task data follwoing the guideline. Uncomment `HOST_PERACT_PACKAGED_DATA_ROOT="/scratch-shared/tmp.lUdVGE8VOd/mixed_onetask_repackaged"` to use the repackaged mixed dataset, which includes both the original RLBench data for the task and DreMa's generated data for the task. Alternatively point `HOST_PERACT_PACKAGED_DATA_ROOT` to where you have placed these data. 

Run the training job with `sbatch snellius_env/3d_diff_training.job`.

To evaluate the trained policy, edit `HOST_CHECKPOINT_FILE_PATH` to point to the saved checkpoint file and run `sbatch snellius_env/3d_diff_eval.job`. This currently does not work as the container does not support CoppeliaSim on Snellius.
