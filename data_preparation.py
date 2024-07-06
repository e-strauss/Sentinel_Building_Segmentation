from time import sleep

import numpy as np
import torch

path = "/Users/eliasstrauss/PycharmProjects/Sentinel_Building_Segmentation/"

import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def extract_patches_img(tensor, patch_size=(128, 128), stride=(64, 64)):
    H, W, C = tensor.shape
    patch_height, patch_width = patch_size
    stride_y, stride_x = stride

    # Calculate the shape and strides for the new view
    new_shape = (
        (H - patch_height) // stride_y + 1,
        (W - patch_width) // stride_x + 1,
        patch_height,
        patch_width,
        C
    )
    new_strides = (
        tensor.strides[0] * stride_y,
        tensor.strides[1] * stride_x,
        tensor.strides[0],
        tensor.strides[1],
        tensor.strides[2]
    )

    patches = np.lib.stride_tricks.as_strided(tensor, shape=new_shape, strides=new_strides)
    return patches


def extract_patches_bld(tensor, patch_size=(128, 128), stride=(64, 64)):
    H, W = tensor.shape
    patch_height, patch_width = patch_size
    stride_y, stride_x = stride

    # Calculate the shape and strides for the new view
    new_shape = (
        (H - patch_height) // stride_y + 1,
        (W - patch_width) // stride_x + 1,
        patch_height,
        patch_width
    )
    new_strides = (
        tensor.strides[0] * stride_y,
        tensor.strides[1] * stride_x,
        tensor.strides[0],
        tensor.strides[1]
    )

    patches = np.lib.stride_tricks.as_strided(tensor, shape=new_shape, strides=new_strides)
    return patches


def merge_patches(cities):
    merged_img_batches = None

    for city in cities[0:2]:
        img = np.load(path + "data/tensors/{}_sentinel.npy".format(city))
        img = extract_patches_img(img)
        img = np.reshape(img, (-1, 128, 128, 4))
        print(city, img.shape)
        if merged_img_batches is None:
            merged_img_batches = img
        else:
            merged_img_batches = np.vstack((merged_img_batches, img))
        del img
    print(merged_img_batches.shape)
    del merged_img_batches

    sleep(5)

    merged_bld_batches = None

    for city in cities[0:2]:
        bld = np.load(path + "data/tensors/{}_building.npy".format(city))
        bld = extract_patches_bld(bld)
        bld = np.reshape(bld, (-1, 128, 128))
        print(city, bld.shape)
        if merged_bld_batches is None:
            merged_bld_batches = bld
        else:
            merged_bld_batches = np.vstack((merged_bld_batches, bld))
        del bld
    print(merged_bld_batches.shape)


def make_batches(cities):
    model = SimpleCNN()
    model.load_state_dict(torch.load("models/cloud_classifier.pth"))
    model.eval()

    for city in cities[3:]:
        img = np.load(path + "data/tensors/{}_sentinel.npy".format(city))
        img = extract_patches_img(img)
        img = np.reshape(img, (-1, 128, 128, 4))

        bld = np.load(path + "data/tensors/{}_building.npy".format(city))
        bld = extract_patches_bld(bld)
        bld = np.reshape(bld, (-1, 128, 128))
        print(city, img.shape, bld.shape)
        batch_size = 128
        indices = []
        for i in range(0, len(img), batch_size):
            tensor_inference = torch.Tensor(img[i:i + batch_size])
            with torch.no_grad():
                cloud_labels = model(tensor_inference.permute(0, 3, 1, 2)).reshape(-1)
            indices_batch = torch.nonzero(cloud_labels < 0.5).reshape(-1)
            indices.append(indices_batch + i)
        indices_flattened = indices[0]
        for idx in indices[1:]:
            indices_flattened = torch.cat((indices_flattened, idx))
        np.save(path + "data/tensors/{}_sentinel_batched_cleaned.npy".format(city), img[indices_flattened])
        np.save(path + "data/tensors/{}_building_batched_cleaned.npy".format(city), bld[indices_flattened])
        print("selected {} samples [{}%]".format(indices_flattened.shape[0],
                                                 (indices_flattened.shape[0] / img.shape[0]) * 100))
        del img, bld


def create_dataset_for_building_segmentation(cities):
    # seed 42 for 100
    # seed 69 for 250
    # seed 69 for 500
    # seed 7 for 1500
    np.random.seed(42)
    nr_samples = 100
    building_dataset = None
    indices = []
    batch_shape = 128
    batch_num_pixel = batch_shape * batch_shape

    # test_data_set has 20% buildings
    minimum_share_of_buildings = {}

    # for larger sample sizes we need to decrease share of buildings for some cities
    if nr_samples == 100:
        minimum_share_of_buildings = {"Dresden": 0.14, "Leipzig": 0.115, "Hamburg": 0.225, "Hannover": 0.185,
                                      "Muenchen": 0.21, "Paris": 0.28, "London": 0.285, "Copenhagen": 0.2,
                                      "Erfurt": 0.11}
    elif nr_samples == 250:
        minimum_share_of_buildings = {"Dresden": 0.1, "Leipzig": 0.08, "Hamburg": 0.185, "Hannover": 0.15,
                                      "Muenchen": 0.175, "Paris": 0.233, "London": 0.249, "Copenhagen": 0.165,
                                      "Erfurt": 0.075}
    elif nr_samples == 500:
        minimum_share_of_buildings = {"Dresden": 0.0875, "Leipzig": 0.06575, "Hamburg": 0.158, "Hannover": 0.1268,
                                      "Muenchen": 0.1485, "Paris": 0.1935, "London": 0.216, "Copenhagen": 0.136,
                                      "Erfurt": 0.0542}
    for city in cities:
        min_share = minimum_share_of_buildings.get(city, 0.2)
        bld = np.load(path + "data/tensors/{}_building_batched_cleaned.npy".format(city))
        print(city, bld.shape[0])
        random_indices = []

        total_num_buildings = batch_num_pixel * 0.2
        count_samples = 1
        iterations = 0
        reset_counter = 0
        while len(random_indices) < nr_samples:
            index = -1
            while index == -1 or index in random_indices:
                index = np.random.randint(0, bld.shape[0])
                iterations += 1
                if iterations > 200000:
                    # reset
                    reset_counter += 1
                    if reset_counter > 15:
                        print("Too many resets for {}!".format(city))
                        exit(1)
                    print("Reset random indices for {}!".format(city))
                    random_indices = []
                    total_num_buildings = batch_num_pixel * 0.2
                    count_samples = 1
                    iterations = 0
            buildings = np.sum(bld[index])
            if (total_num_buildings / batch_num_pixel / count_samples < min_share * 0.95
                    and buildings < batch_num_pixel * min_share * 1.1):
                continue
            if count_samples % int(nr_samples * 0.1) == 0:
                print(count_samples)
            random_indices.append(index)
            total_num_buildings += buildings
            count_samples += 1
        print(total_num_buildings / batch_num_pixel / nr_samples)
        indices.append(random_indices)
        bld = bld[random_indices]
        if building_dataset is None:
            building_dataset = bld.copy()
        else:
            building_dataset = np.concat((building_dataset, bld), axis=0)
        del bld
    print("buildings: {}%".format(np.sum(building_dataset) / building_dataset.size))
    np.save(path + "data/tensors/building_segmentation_labels_dataset_equal_provenance{}.npy".format(nr_samples),
            building_dataset)
    del building_dataset

    image_dataset = None
    for random_indices, city in zip(indices, cities):
        img = np.load(path + "data/tensors/{}_sentinel_batched_cleaned.npy".format(city))
        img = img[random_indices]
        if image_dataset is None:
            image_dataset = img.copy()
        else:
            image_dataset = np.concat((image_dataset, img), axis=0)
        del img
    np.save(path + "data/tensors/building_segmentation_images_dataset_equal_provenance{}.npy".format(nr_samples),
            image_dataset)
