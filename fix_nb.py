#not important, can be remove?

import json
import os

with open('main.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            if 'images = sorted(os.listdir(clean_dir))' in line:
                new_source.append(line)
                new_source.append('    clean_folder = os.path.basename(os.path.normpath(clean_dir))\n')
            elif './datasets/denoising_datasets/' in line and 'rel_noisy =' in line:
                new_source.append(line.replace('./datasets/denoising_datasets', './denoising_datasets'))
            elif './datasets/denoising_datasets/' in line and 'rel_clean =' in line:
                new_source.append(line.replace('./datasets/denoising_datasets', './denoising_datasets').replace('clean', '{clean_folder}'))
            elif 'rel_clean = f"./denoising_datasets/' in line:
                new_source.append(line.replace('clean', '{clean_folder}'))
            else:
                new_source.append(line)
        cell['source'] = new_source

with open('main.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
