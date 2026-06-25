import json
import os
import random

import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm


def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1,
    )
    return image.crop(random_region), label.crop(random_region)


def randomRotation(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomPeper(img):
    img = np.array(img)
    noise_num = int(0.0015 * img.shape[0] * img.shape[1])
    for _ in range(noise_num):
        rand_x = random.randint(0, img.shape[0] - 1)
        rand_y = random.randint(0, img.shape[1] - 1)
        img[rand_x, rand_y] = 0 if random.randint(0, 1) == 0 else 255
    return Image.fromarray(img)


def split_ref_data(data_root, record_file='./data/refsplits1.json'):
    assert os.path.exists(data_root)
    os.makedirs('/'.join(record_file.split('/')[:-1]), exist_ok=True)

    refsplits = {'train': {}, 'test': {}}
    ref_image_root = os.path.join(data_root, 'Ref', 'Images')
    assert os.path.exists(ref_image_root)

    for cate in os.listdir(ref_image_root):
        ref_cate_image_dir = os.path.join(data_root, 'Ref', 'Images', cate)
        ref_cate_image_names = os.listdir(ref_cate_image_dir)
        assert len(ref_cate_image_names) == 25
        random.shuffle(ref_cate_image_names)
        refsplits['train'][cate] = [name[:-4] for name in ref_cate_image_names[:20]]
        refsplits['test'][cate] = [name[:-4] for name in ref_cate_image_names[20:]]

    with open(record_file, 'w') as f:
        json.dump(refsplits, f, indent=4)


def collect_r2c_data(data_root, mode='train', record_file='./data/refsplits1.json'):
    if not os.path.exists(record_file):
        split_ref_data(data_root, record_file)

    assert os.path.exists(data_root)
    camo_imgs_dir = os.path.join(data_root, 'Camo', mode if mode != 'val' else 'test', 'Imgs')
    camo_gts_dir = os.path.join(data_root, 'Camo', mode if mode != 'val' else 'test', 'GT')
    ref_feats_dir = os.path.join(data_root, 'Ref', 'pro_2048')
    assert os.path.exists(camo_imgs_dir) and os.path.exists(camo_gts_dir) and os.path.exists(ref_feats_dir)

    camo_classes = os.listdir(camo_imgs_dir)
    ref_classes = os.listdir(ref_feats_dir)
    assert len(camo_classes) == len(ref_classes)

    with open(record_file, 'r') as f:
        splits = json.load(f)

    image_label_list = []
    class_file_list = {}
    split_key = mode if mode != 'val' else 'test'

    for cate in tqdm(camo_classes):
        camo_cate_imgs_dir = os.path.join(camo_imgs_dir, cate)
        camo_cate_gts_dir = os.path.join(camo_gts_dir, cate)
        camo_img_names = sorted(os.listdir(camo_cate_imgs_dir))
        camo_gt_names = sorted(os.listdir(camo_cate_gts_dir))
        assert len(camo_img_names) == len(camo_gt_names)

        image_label_list.extend([
            (
                os.path.join(camo_cate_imgs_dir, camo_img_names[idx]),
                os.path.join(camo_cate_gts_dir, camo_gt_names[idx]),
            )
            for idx in range(len(camo_img_names))
        ])

        ref_cate_feats_dir = os.path.join(ref_feats_dir, cate)
        ref_cate_split_names = splits[split_key][cate]
        class_file_list[cate] = [
            os.path.join(ref_cate_feats_dir, f'{sample_name}.npy')
            for sample_name in ref_cate_split_names
        ]

    print(f'>>> {mode}ing with {len(image_label_list)} r2c samples')
    return image_label_list, class_file_list
