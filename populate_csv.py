# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pickle
import argparse
import pandas as pd
import os
import yaml
import datetime
from tqdm import tqdm

def pre_populate_csv(data_set, path_to_generations):
    if data_set == 'dollarstreet':
        config_data_path='config_dstreet.yaml'
        with open('dollarstreet_counts_per_obj_region.pkl', 'rb') as f:
            dollarstreet_counts_per_obj_region = pickle.load(f)

    elif data_set == 'geode':
        config_data_path='config_geode.yaml'

    with open(config_data_path, 'r') as config_file:
        config_data = yaml.safe_load(config_file)
    list_vscore_scales = config_data['list_vscore_scales']

    now = datetime.datetime.now()
    root_dir = path_to_generations + str(now.strftime("%Y-%m-%d-%H-%M-%S-%f")).replace('-', '_')
    os.mkdir(root_dir)
    config_data['root_dir'] = root_dir

    with open(f"{root_dir}/config.yaml", "w") as f:
        yaml.dump(config_data, f)

    for starting_seed in config_data['list_starting_seed']:
        for vscore_scale in tqdm(list_vscore_scales):
            for cfg in config_data['list_cfgs']:
                for region in config_data['list_regions']:
                    data = []
                    img_id = 0
                    seed = starting_seed
                    for object_name in config_data['objects']:
                        prompt = ''
                        if config_data['obj_in_prompt']:
                            prompt = prompt + object_name
                        if config_data['region_in_prompt']:
                            prompt = prompt + f' in {region}'

                        count_per_obj = config_data['count_per_obj']
                        if count_per_obj is None and data_set == 'dollarstreet':
                            count_per_obj = dollarstreet_counts_per_obj_region[f'{object_name}_{region}']

                        for _ in range(count_per_obj):
                            row = {
                                'prompt': prompt,
                                'object': object_name,
                                'region': region if (config_data['region_in_prompt'] or config_data['guidance_type'] in ['clip_loss', 'clip_entropy', 'vscore']) else None,
                                'img_id': img_id,
                                'seed': seed,
                                'vscore_scale': vscore_scale,
                                'cfg': cfg,
                                'guidance_type': config_data['guidance_type'],
                                'num_inference_steps': config_data['num_inference_steps'],
                                'guidance_freq': config_data['guidance_freq'],
                                'beta': config_data['beta'],
                            }
                            data.append(row)
                            img_id += 1
                            seed += 1

                    df = pd.DataFrame(data)
                    csv_file_name = f'{region}_{vscore_scale}_{cfg}_{starting_seed}.csv'
                    csv_file_path = f'{root_dir}/{csv_file_name}'
                    df.to_csv(csv_file_path, index=False)
    print(f'CSV files created at root directory: {root_dir}')
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=str, default='geode')
    parser.add_argument('--path_to_generations', type=str, default='generated_datasets/')
    args = parser.parse_args()
    if not os.path.exists(args.path_to_generations):
        os.mkdir(args.path_to_generations)
    pre_populate_csv(args.data_set, args.path_to_generations)
    
