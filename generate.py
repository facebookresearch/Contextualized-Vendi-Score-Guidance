# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import random
import csv
import argparse
import pandas as pd
from PIL import Image
import pickle

import torch
from diffusers import DiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPModel
import clip

import submitit
from submitit.helpers import Checkpointable, DelayedSubmission

def get_F_M(M, F, f):
        F_ = torch.cat((F, f))
        m = torch.mm(F_, f.T)
        M_ = torch.cat((M, m[:-1].T))
        M_ = torch.cat((M_, m), dim=1)
        return F_, M_


def precompute_F_M_real(pre_process_clip, clip_for_guidance, dataset, num_real_samples):
    # precomputations of the VS with respect to examplar images.
    F_M_real_all = {}
    if dataset == 'dollarstreet':
        with open('dollar_street_train.pkl', 'rb') as f:
            dollar_street_train = pickle.load(f)
        all_objects_ds = list(set(dollar_street_train['object_reformatted'].tolist()))
        for obj in all_objects_ds:
            selected_samples_per_obj_list = dollar_street_train[dollar_street_train['object_reformatted'] == obj]['file_path'].tolist()
            root_dir = 'PATH_TO_DOLLARSTREET'
            F_M_real = None
            for obj_path in selected_samples_per_obj_list:
                img_path = os.path.join(root_dir,obj_path)
                image = pre_process_clip(Image.open(img_path)).unsqueeze(0).to('cuda')
                features_ = clip_for_guidance.encode_image(image).to(torch.float32)
                features_ = features_ / features_.norm(2, dim=1, keepdim=True)
                if F_M_real is None:
                    F_M_real = [features_.detach(), torch.mm(features_, features_.T)]
                else:
                    F_M_real = get_F_M(M=F_M_real[1], F=F_M_real[0], f=features_.detach())
            assert F_M_real[1].shape[0] == 4
            assert F_M_real[1].shape[1] == 4
            assert F_M_real[0].shape[0] == 4
            assert F_M_real[0].shape[1] == 512
            F_M_real_all[obj] = (F_M_real, selected_samples_per_obj_list)
            print(f'obj {obj} done!')

    if dataset == 'geode':
        root_dir = 'PATH_TO_GEODE'
        real_hold_not_in_eval_sub = pd.read_csv('real_hold_not_in_eval_sub.csv')
        # real_hold_not_in_eval_sub.csv is a csv of the images that are not used in evaluation
        # for quick test of the code you can use the whole dataset but for general use of the code
        # you will need to define an evaluation set.

        all_objects = list(set(real_hold_not_in_eval_sub['object'].tolist()))
        for obj in all_objects:
            grouped = real_hold_not_in_eval_sub.groupby('object')
            # change random_state to get a new set of samples
            # change n to get a different sample size
            selected_samples = grouped.apply(lambda x: x.sample(n=num_real_samples, random_state=10))
            selected_samples_per_obj_list = selected_samples[selected_samples['object'] == obj]['file_path'].tolist()
            F_M_real = None
            for obj_path in selected_samples_per_obj_list:
                img_path = os.path.join(root_dir, 'images',obj_path)
                image = pre_process_clip(Image.open(img_path)).unsqueeze(0).to('cuda')
                features_ = clip_for_guidance.encode_image(image).to(torch.float32)
                features_ = features_ / features_.norm(2, dim=1, keepdim=True)
                if F_M_real is None:
                    F_M_real = [features_.detach(), torch.mm(features_, features_.T)]
                else:
                    F_M_real = get_F_M(M=F_M_real[1], F=F_M_real[0], f=features_.detach())

            assert F_M_real[1].shape[0] == num_real_samples
            assert F_M_real[1].shape[1] == num_real_samples
            assert F_M_real[0].shape[0] == num_real_samples
            assert F_M_real[0].shape[1] == 512
            F_M_real_all[obj] = (F_M_real, selected_samples_per_obj_list)
            print(f'obj {obj} done!')

    with open(f'F_M_real_all_{dataset}_{num_real_samples}.pkl', 'wb') as f:
        pickle.dump(F_M_real_all, f)
    print(f'F_M_real_all_{dataset}_{num_real_samples}.pkl saved!')


def generate(dfs_n_csv_path, clip_for_guidance, pre_process_clip, clip_model, feature_extractor, num_real_samples, dataset):
    torch_dtype = torch.float16
    criteria_pipe = DiffusionPipeline.from_pretrained(
                "PATH_TO_YOUR_LDM"
                custom_pipeline="criteria_guided_sampling.py",
                clip_model=clip_model,
                feature_extractor=feature_extractor,
                torch_dtype=torch_dtype,
            )
    criteria_pipe = criteria_pipe.to("cuda")

    if dataset == 'geode':
        regions_list = ['Africa', 'WestAsia', 'EastAsia', 'SouthEastAsia', 'Americas', 'Europe']
    elif dataset == 'dollarstreet':
        regions_list = ['Europe', 'Africa', 'the Americas', 'Asia']
    else:
        raise NotImplementedError

    if not os.path.exists(f'F_M_real_all_{dataset}_{num_real_samples}.pkl'):
        precompute_F_M_real(pre_process_clip, clip_for_guidance, dataset, num_real_samples)
    with open(f'F_M_real_all_{dataset}_{num_real_samples}.pkl', 'rb') as f:
        F_M_real_all = pickle.load(f)

    df, csv_path = dfs_n_csv_path
    obj_ref = ''
    F_M = None
    F_M_real = None
    for index, row in df.iterrows():
        guidance_type = row['guidance_type']
        prompt = row['prompt']
        obj = row['object']
        region = row['region']
        img_id = row['img_id']
        guidance_freq = row['guidance_freq']
        beta = row['beta']
        if obj_ref != obj and guidance_type == 'vscore_clip':
            obj_ref = obj
            F_M = None
            F_M_real = F_M_real_all[obj][0]
            print('---------------vscore bank is initiated!-----------')
        seed = row['seed']
        vscore_scale = row['vscore_scale']
        cfg = row['cfg']
        num_inference_steps = row['num_inference_steps']
        folder_name = csv_path.split('.csv')[0]
        if not os.path.exists(folder_name):
            os.mkdir(f'{folder_name}')
        image_path = f'{folder_name}/{prompt}__{img_id}.png'
        if not os.path.exists(image_path):
            generator = torch.Generator(device='cuda')
            generator.manual_seed(seed)
            import time
            t1 = time.time()
            if guidance_type in ['clip_entropy', 'clip_loss', 'vscore_clip']:
                out, F_M = criteria_pipe(
                    prompt=prompt,
                    guidance_scale=cfg,
                    criteria=guidance_type,
                    criteria_guidance_scale=vscore_scale,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    clip_for_guidance=clip_for_guidance,
                    guidance_freq=guidance_freq,
                    region=region,
                    F_M=F_M,
                    F_M_real=F_M_real,
                    beta=beta,
                    regions_list=regions_list)
                image = out[0]
            else:
                raise NotImplementedError
            image.save(image_path)
            print(image_path)
        else:
            print(f'image id {img_id} is already generated!')

class Runner(Checkpointable):

    def __init__(self, dfs_n_csv_path,  clip_for_guidance, pre_process_clip,
                 clip_model, feature_extractor, num_real_samples, dataset):
        self.dfs_n_csv_path = dfs_n_csv_path
        self.clip_for_guidance = clip_for_guidance
        self. clip_model =  clip_model
        self.feature_extractor = feature_extractor
        self.pre_process_clip = pre_process_clip
        self.num_real_samples = num_real_samples
        self.dataset = dataset

    def __call__(self):
        generate(self.dfs_n_csv_path, self.clip_for_guidance,
                self.pre_process_clip, self.clip_model,
                self.feature_extractor, self.num_real_samples,
                self.dataset)


    def checkpoint(self):
        print(f"Requeuing task {self.task_id}")
        return DelayedSubmission(self)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit_it', action='store_true', default=False, help='submit to cluster')
    parser.add_argument('--slurm_partition', type=str, default='SLURM_PARTITION')
    parser.add_argument('--root_dir', type=str, default='generated_datasets/2024_07_08_19_28_59_179592',
                        help='root dir that contains all the csv files of samples to be generated.')
    parser.add_argument('--chunk_size', type=int, default=180, help='the number of images passed to each gpu.')
    parser.add_argument('--num_real_samples', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='geode')

    args = parser.parse_args()


    torch_dtype = torch.float16

    feature_extractor = CLIPFeatureExtractor.from_pretrained("PATH_TO_CLIP_MODEL")
    clip_model = CLIPModel.from_pretrained("PATH_TO_CLIP_MODEL", torch_dtype=torch_dtype)
    clip_for_guidance, pre_process_clip = clip.load("ViT-B/32", device='cuda')


    if args.chunk_size % 180 != 0 and args.dataset == 'geode':
        print('chunk_size has to be divisible by 180 (count_per_obj)!')
        raise NotImplementedError
    root_dir = args.root_dir
    list_dfs_n_csv_path = []
    for file in os.listdir(root_dir):
        if '.csv' in file:
            csv_path = root_dir + '/' + file
            df = pd.read_csv(csv_path)
            for chunk in range(0, len(df), args.chunk_size):
                list_dfs_n_csv_path += [(df.iloc[chunk:chunk + args.chunk_size], csv_path)]

    if not args.submit_it:
        for dfs_n_csv_path in list_dfs_n_csv_path:
            Runner(dfs_n_csv_path, clip_for_guidance,
                   pre_process_clip, clip_model, feature_extractor,
                   args.num_real_samples, args.dataset)()
    else:
        submitit_path = './submitit'
        executor = submitit.AutoExecutor(folder=submitit_path, slurm_max_num_timeout=30)
        executor.update_parameters(
            gpus_per_node=1, array_parallelism=512,
            tasks_per_node=1, cpus_per_task=1, nodes=1,
            timeout_min=2000,
            slurm_partition=args.slurm_partition,
            slurm_signal_delay_s=120)
        jobs = []
        with executor.batch():
            for dfs_n_csv_path in list_dfs_n_csv_path:
                runner = Runner(dfs_n_csv_path, clip_for_guidance,
                                pre_process_clip, clip_model,
                                feature_extractor, args.num_real_samples,
                                args.dataset)
                job = executor.submit(runner)
                jobs.append(job)
