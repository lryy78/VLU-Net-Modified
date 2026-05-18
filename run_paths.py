import os
import numpy as np
from PIL import Image

# =========================================================
# DERAINING
# =========================================================

# from IPython.core import display_functions
input_dir = "./datasets/deraining_datasets/Rain100L/rainy"
gt_dir = "./datasets/deraining_datasets/Rain100L/gt"

relative_input = "./deraining_datasets/Rain100L/rainy"
relative_gt = "./deraining_datasets/Rain100L/gt"

for txt_name in ["train_paths.txt", "test_paths.txt"]:

    output_txt = f"./datasets/deraining_datasets/Rain100L/{txt_name}"

    inputs = sorted(os.listdir(input_dir))

    with open(output_txt, "w") as f:
        for name in inputs:

            input_path = os.path.join(relative_input, name)
            gt_path = os.path.join(relative_gt, name)

            if os.path.exists(os.path.join(gt_dir, name)):
                f.write(f"{input_path},{gt_path}\n")


# =========================================================
# DEBLURRING
# =========================================================

input_dir = "./datasets/deblurring_datasets/GoPro/blurry"
gt_dir = "./datasets/deblurring_datasets/GoPro/sharp"

relative_input = "./deblurring_datasets/GoPro/blurry"
relative_gt = "./deblurring_datasets/GoPro/sharp"

for txt_name in ["train_paths.txt", "test_paths.txt"]:

    output_txt = f"./datasets/deblurring_datasets/GoPro/{txt_name}"

    inputs = sorted(os.listdir(input_dir))

    with open(output_txt, "w") as f:
        for name in inputs:

            input_path = os.path.join(relative_input, name)
            gt_path = os.path.join(relative_gt, name)

            if os.path.exists(os.path.join(gt_dir, name)):
                f.write(f"{input_path},{gt_path}\n")


# =========================================================
# DEHAZING
# =========================================================

hazy_dir = "./datasets/dehazing_datasets/SOTS_outdoors/hazy"
gt_dir = "./datasets/dehazing_datasets/SOTS_outdoors/gt"

relative_hazy = "./dehazing_datasets/SOTS_outdoors/hazy"
relative_gt = "./dehazing_datasets/SOTS_outdoors/gt"

for txt_name in ["train_paths.txt", "test_paths.txt"]:

    output_txt = f"./datasets/dehazing_datasets/{txt_name}"

    hazy_images = sorted(os.listdir(hazy_dir))

    with open(output_txt, "w") as f:
        for img in hazy_images:

            hazy_path = os.path.join(relative_hazy, img)

            base = img.split("_")[0]

            gt_filename = base + ".png"
            gt_path = os.path.join(relative_gt, gt_filename)

            if os.path.exists(os.path.join(gt_dir, gt_filename)):
                f.write(f"{hazy_path},{gt_path}\n")


# =========================================================
# LOWLIGHT TRAIN
# =========================================================

low_dir = "./datasets/delowlight_datasets/LoL/our485/low"
high_dir = "./datasets/delowlight_datasets/LoL/our485/high"

relative_low = "./delowlight_datasets/LoL/our485/low"
relative_high = "./delowlight_datasets/LoL/our485/high"

output_txt = "./datasets/delowlight_datasets/LoL/train_paths.txt"

files = sorted(os.listdir(low_dir))

with open(output_txt, "w") as f:
    for name in files:

        low_path = os.path.join(relative_low, name)
        high_path = os.path.join(relative_high, name)

        f.write(f"{low_path},{high_path}\n")


# =========================================================
# LOWLIGHT TEST
# =========================================================

low_dir = "./datasets/delowlight_datasets/LoL/eval15/low"
high_dir = "./datasets/delowlight_datasets/LoL/eval15/high"

relative_low = "./delowlight_datasets/LoL/eval15/low"
relative_high = "./delowlight_datasets/LoL/eval15/high"

output_txt = "./datasets/delowlight_datasets/LoL/test_paths.txt"

files = sorted(os.listdir(low_dir))

with open(output_txt, "w") as f:
    for name in files:

        low_path = os.path.join(relative_low, name)
        high_path = os.path.join(relative_high, name)

        f.write(f"{low_path},{high_path}\n")



def add_gaussian_noise(img, sigma):
    img = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma/255.0, img.shape)
    noisy = np.clip(img + noise, 0, 1)
    return (noisy * 255).astype(np.uint8)


def generate_dataset(clean_dir, base_out, dataset_name):
    sigmas = [15, 25, 50]
    images = sorted(os.listdir(clean_dir))
    clean_folder = os.path.basename(os.path.normpath(clean_dir))

    for sigma in sigmas:
        noisy_dir = os.path.join(base_out, f"noisy{sigma}")
        os.makedirs(noisy_dir, exist_ok=True)

        txt_path = os.path.join(base_out, f"{sigma}_test_paths.txt")

        with open(txt_path, "w") as f:
            for img_name in images:

                clean_path = os.path.join(clean_dir, img_name)

                img = Image.open(clean_path).convert("RGB")
                noisy = add_gaussian_noise(img, sigma)

                noisy_path = os.path.join(noisy_dir, img_name)
                Image.fromarray(noisy).save(noisy_path)

                rel_noisy = f"./denoising_datasets/{dataset_name}/noisy{sigma}/{img_name}"
                rel_clean = f"./denoising_datasets/{dataset_name}/{clean_folder}/{img_name}"

                f.write(f"{rel_noisy},{rel_clean}\n")

    rand_txt = os.path.join(base_out, "rand_test_paths.txt")
    rand_dir = os.path.join(base_out, "noisy_rand")
    os.makedirs(rand_dir, exist_ok=True)

    with open(rand_txt, "w") as f:
        for img_name in images:

            sigma = np.random.choice(sigmas)

            clean_path = os.path.join(clean_dir, img_name)
            img = Image.open(clean_path).convert("RGB")
            noisy = add_gaussian_noise(img, sigma)

            noisy_path = os.path.join(rand_dir, img_name)
            Image.fromarray(noisy).save(noisy_path)

            rel_noisy = f"./denoising_datasets/{dataset_name}/noisy_rand/{img_name}"
            rel_clean = f"./denoising_datasets/{dataset_name}/{clean_folder}/{img_name}"

            f.write(f"{rel_noisy},{rel_clean}\n")
generate_dataset(
    "./datasets/denoising_datasets/CBSD68/original_png",
    "./datasets/denoising_datasets/CBSD68",
    "CBSD68"
)
generate_dataset(
    "./datasets/denoising_datasets/Urban100_HR/clean",
    "./datasets/denoising_datasets/Urban100_HR",
    "Urban100_HR"
)
