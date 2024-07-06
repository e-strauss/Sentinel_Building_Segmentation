from matplotlib import pyplot as plt
from pyrosm import get_data, OSM
import openeo
import rasterio
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.features import rasterize

path = "/Users/eliasstrauss/PycharmProjects/Sentinel_Building_Segmentation/"


def create_buildings_map(city):
    fp = get_data(city)
    osm = OSM(fp)
    buildings = osm.get_buildings()
    buildings.to_file(path + "data/buildings/" + city + "_buildings.gpkg", driver="GPKG")
    bounds = buildings.total_bounds
    return bounds


def create_sentinel_map(city, coords, temporal_extent=["2024-04-01", "2024-07-01"], max_cloud_cover=20):
    print(city)
    print(coords)

    connection = openeo.connect("https://openeo.dataspace.copernicus.eu/")
    connection.authenticate_oidc()
    datacube = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=coords,
        temporal_extent=temporal_extent,
        bands=["B02", "B03", "B04", "B08"],
        max_cloud_cover=max_cloud_cover
    )
    reduced_datacube = datacube.max_time()
    exported_path = path + "/data/satellite/{}.tiff".format(city)
    result = reduced_datacube.save_result(format="GTiff")
    job = result.create_job()
    job.start_and_wait()
    job.get_results().download_files(target=exported_path)
    return exported_path


def read_coords_and_get_sentinel_satellite_images():
    with open(path + "data/city_coords.txt", "r") as f:
        lines = f.readlines()
        a = lines[1]
        a = a.replace("[", '').replace("]", '')
        a = a.split(' ')
        city = a[0]
        b = ["west", "south", "east", "north"]
        coords = {bi: float(a[i + 1]) for i, bi in enumerate(b)}
        create_sentinel_map(city, coords)


def align_buildings_to_sentinel(city):
    tiff = rasterio.open(path + "data/satellite/{}.tiff/openEO.tif".format(city))
    image_tensor = tiff.read([1, 2, 3, 4])

    # remove int16 overflow values by replacing them by the average over all RGB values
    cleaned_avg = np.mean(image_tensor[image_tensor > 0])
    image_tensor[image_tensor < 0] = cleaned_avg

    # normalize data between 0:1
    min_val = np.reshape(np.min(image_tensor, axis=(1, 2)), (-1, 1, 1))
    max_val = np.reshape(np.max(image_tensor, axis=(1, 2)), (-1, 1, 1))
    image_tensor = (image_tensor - min_val) / (max_val / 2 - min_val)
    image_tensor[image_tensor > 1.0] = 1.0
    # CHW -> HWC
    image_tensor = np.transpose(image_tensor, (1, 2, 0))

    # project to same CRS
    df = gpd.read_file(path + "data/buildings/{}_buildings.gpkg".format(city))
    df = df.to_crs(tiff.crs)
    building_matrix = rasterize(df["geometry"], out_shape=tiff.shape, transform=tiff.transform)

    assert building_matrix.shape == image_tensor.shape[:2]
    return image_tensor, building_matrix


def plot_city(city, img, bld, shape=(3, 3), dpi=300):
    height, width = bld.shape
    s_height, s_width = shape[0] * dpi, shape[1] * dpi
    offset_height = int((height - s_height) / 2)
    offset_width = int((width - s_width) / 2)

    s_img = img[offset_height:offset_height + s_height, offset_width:offset_width + s_width]
    s_bld = bld[offset_height:offset_height + s_height, offset_width:offset_width + s_width]

    # single band blue
    fig, ax = plt.subplots(figsize=(shape[1] * 3, shape[0] * 3))
    ax.imshow(s_img[:, :, 2], cmap="gray")
    fig.savefig(path + "plots/{}_single_blue.pdf".format(city), bbox_inches='tight', dpi=dpi)

    # single band ir
    fig, ax = plt.subplots(figsize=(shape[1] * 3, shape[0] * 3))
    ax.imshow(s_img[:, :, 3], cmap="gray")
    fig.savefig(path + "plots/{}_single_ir.pdf".format(city), bbox_inches='tight', dpi=dpi)

    # RGB
    fig, ax = plt.subplots(figsize=(shape[1] * 3, shape[0] * 3))
    ax.imshow(s_img[:, :, :3])
    fig.savefig(path + "plots/{}_sentinel.pdf".format(city), bbox_inches='tight', dpi=dpi)

    # IRB
    fig, ax = plt.subplots(figsize=(shape[1] * 3, shape[0] * 3))
    ax.imshow(s_img[:, :, [3, 0, 2]])
    fig.savefig(path + "plots/{}_irb.pdf".format(city), bbox_inches='tight', dpi=dpi)

    # Buildings
    fig, ax = plt.subplots(figsize=(shape[1] * 3, shape[0] * 3))
    ax.imshow(s_bld, cmap='Blues')
    fig.savefig(path + "plots/{}_buildings.pdf".format(city), bbox_inches='tight', dpi=dpi)

    # overlapped
    overlapped = s_img[:, :, :3]
    overlapped[s_bld == 1] = [0, 0, 1]
    fig, ax = plt.subplots(figsize=(shape[1] * 3, shape[0] * 3))
    ax.imshow(overlapped)
    fig.savefig(path + "plots/{}_overlapped.pdf".format(city), bbox_inches='tight', dpi=dpi)


def data_acquisition_and_alignment_pipeline(cities):
    with open(path + "data/city_coords.txt", 'r') as f:
        lines = f.readlines()
    lines = [line.replace("[", '').replace("]", '') for line in lines]
    existing_building_data = [line.split(" ") for line in lines]
    existing_building_data = {line[0]: [a for a in line[1:] if len(a) > 0] for line in existing_building_data}
    with open(path + "data/city_coords.txt", 'a') as f:
        for city in cities:
            if city in existing_building_data:
                print("found existing building data for {}".format(city))
                a = existing_building_data[city]
            else:
                a = create_buildings_map(city)
                f.write(city + " " + str(a) + "\n")
            b = ["west", "south", "east", "north"]
            coords = {bi: float(a[i]) for i, bi in enumerate(b)}
            create_sentinel_map(city, coords)

            img, bld = align_buildings_to_sentinel(city)

            # img = np.reshape(img, (img.shape[0], -1))
            np.save(path + "data/tensors/{}_sentinel.npy".format(city), img)
            np.save(path + "data/tensors/{}_building.npy".format(city), bld)

            plot_city(city, img, bld)


def data_alignment_and_save_pipeline(cities):
    for city in cities:
        img, bld = align_buildings_to_sentinel(city)

        # img = np.reshape(img, (img.shape[0], -1))
        np.save(path + "data/tensors/{}_sentinel.npy".format(city), img)
        np.save(path + "data/tensors/{}_building.npy".format(city), bld)


def get_test_data_of_berlin():
    city = "Berlin"
    a = [13.294333, 52.454927, 13.500205, 52.574409]
    b = ["west", "south", "east", "north"]
    coords = {bi: float(a[i]) for i, bi in enumerate(b)}
    # create_sentinel_map(city, coords, ["2024-06-26", "2024-06-27"], 10)
    # create_buildings_map(city)
    img, bld = align_buildings_to_sentinel(city)

    # RGB
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(img[:, :, :3])
    fig.savefig(path + "plots/test_set_{}_sentinel.pdf".format(city), bbox_inches='tight', dpi=200)

    # overlapped
    overlapped = img[:, :, :3]
    overlapped[bld == 1] = [0, 0, 1]
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(overlapped)
    fig.savefig(path + "plots/test_set_{}_overlapped.pdf".format(city), bbox_inches='tight', dpi=200)

    np.save(path + "data/tensors/test_set_{}_sentinel.npy".format(city), img)
    np.save(path + "data/tensors/test_set_{}_building.npy".format(city), bld)
