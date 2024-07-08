# [Improving Geo-diversity of Generated Images with Contextualized Vendi Score Guidance (ECCV 2024 accepted paper)](https://arxiv.org/abs/2406.04551)

[![arXiv](https://img.shields.io/badge/arXiv-2406.04551-b31b1b.svg)](https://arxiv.org/abs/2406.04551)

> **Improving Geo-diversity of Generated Images with Contextualized Vendi Score Guidance**<br>
> Reyhane Askari Hemmat*<sup>1</sup>, Melissa Hall *<sup>1</sup>, Alicia Sun<sup>1</sup>, Candace Ross<sup>1</sup>, Michal Drozdzal<sup>1</sup>, Adriana Romero-Soriano<sup>1, 2, 3, 4</sup>

> <sup>1</sup> FAIR at Meta, <sup>2</sup> Mila, <sup>3</sup> McGill University, <sup>4</sup> Canada CIFAR AI chair

>**Abstract**: <br>
> With the growing popularity of text-to-image generative models, there has been increasing focus on understanding their risks and biases. Recent work has found that state-of-the-art models struggle to depict everyday objects with the true diversity of the real world and have notable gaps between geographic regions. In this work, we aim to increase the diversity of generated images of common objects such that per-region variations are representative of the real world. We introduce an inferencetime intervention, contextualized Vendi Score Guidance (c-VSG), that guides the backwards steps of latent diffusion models to increase the diversity of a sample as compared to a “memory bank” of previously generated images while constraining the amount of variation within that of an exemplar set of real-world contextualizing images. We evaluate c-VSG with two geographically representative datasets and find that it substantially increases the diversity of generated images, both for the worstperforming regions and on average, while simultaneously maintaining or improving image quality and consistency. Additionally, qualitative analyses reveal that diversity of generated images is significantly improved, including along the lines of reductive region portrayals present in the original model. We hope that this work is a step towards text-to-image generative models that reflect the true geographic diversity of the world.

## Description
This repo contains the official code for the contextualized Vendi Score Guidance (c-VSG) method. Our code builds on, and shares requirements with [DIG-In](https://github.com/facebookresearch/DIG-In/tree/main). To run the evaluations, please use the DIG-In codebase.

## Usage
This codebase generates samples of an LDM using c-VSG method in large scale and saves the generations in a systematic way. The samples are then used to compute the [DIG-In](https://github.com/facebookresearch/DIG-In/tree/main) metrics. We first generate a set of csv files that contain the information of the samples that need to be generated. Then the csv files are passed to a generation code which generates each sample according to the hyper-parameters defined in the csv file and saves them. Then the genrations are used to compute the DIG-In metrics.

We support two datasets, [GeoDe](https://geodiverse-data-collection.cs.princeton.edu/) and [DollarStreet](https://mlcommons.org/datasets/dollar-street/).

### Creating csv files of generations
As a first step, defining the hyper-parameters in the config.yaml files of the dataset: Each config.yaml corresponds to one set of experiments, we define all the hyper-parameters of generations in this yaml file.

As a second step, we populate the csv files given the config file: 

```
python populate_csv.py --data_set geode --path_to_generations PATH_TO_GENERATION
```

This script reads the hyper-parameters of the experiment and populates many csv files corresponding to each region and set of hyper-parameters. Each csv file is stored in a ROOT_DIR which corresponds to the time of day to avoid over-writing experiments (this is printed in the terminal). In the ROOT_DIR all the data related to this experiment are stored. Including a copy of the config.yaml file. Specifically, at this stage a csv file corresponding to '{region}{vscore_scale}{cfg}_{starting_seed}.csv' is created in the ROOT_DIR.


### Generation
Given a root file, the generation code reads all the csv files in that root directory and generates the samples. The code supports generating on multiple gpus in parallel. There is no need for the gpus to be on the same node.

```
python generate.py --submit_it --root_dir ROOT_DIR
```

The samples are saved'ROOT_DIR/{region}{edit_guidance_scale}{clip_guidance_scale}_{starting_seed}'.

This generation code, runs a custom pipe for a predefined latent diffusion model. We implement Algorithm 1 in the paper in the `criteria_guided_sampling.py` file in the `cond_fn` function. We have implemeneted a very efficient algorithm that stores the pre-computed values of the similarity matrix in the `F_M` variable and passes this variable to be used for the generation of the next image.

### Computing metrics
Use the DIG-In evaluation metrics to compute the metrics over the regions.

## Tips and Tricks
- One can use this code with any pre-trained latent diffusion model. You will need to fix some imports based on your preference of the LDM model selected from the hugging face library. Specifically, in the generate.py code, you need to define `PATH_TO_YOUR_LDM` and `PATH_TO_CLIP_MODEL` variables. Furthermore, for the datasets, the paths should be updated in the variable `PATH_TO_DOLLARSTREET` or `PATH_TO_GEODE`.

## License 
This code is licensed under CC-BY-NC license.


## Citation

If you make use of our work or code, please cite our paper:
```
@article{askari2024improving,
  title={Improving Geo-diversity of Generated Images with Contextualized Vendi Score Guidance},
  author={Askari Hemmat, Reyhane and Hall, Melissa and Sun, Alicia and Ross, Candace and Drozdzal, Michal and Romero-Soriano, Adriana},
  journal={arXiv e-prints},
  pages={arXiv--2406},
  year={2024}
}
```
