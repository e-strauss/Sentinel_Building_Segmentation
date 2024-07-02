import openeo


def create_sentinel_map(coords=None):
    connection = openeo.connect("https://openeo.dataspace.copernicus.eu/")
    connection.authenticate_oidc()
    datacube = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent={"west": 12.02000046, "south": 50.97999954, "east": 12.91999912, "north": 51.59000015},
        temporal_extent=["2023-03-01", "2023-03-01"],
        bands=["B04"]
    )
    reduced_datacube = datacube.reduce_dimension(dimension="t", reducer="mean")
    exported_path = "data/reduced_b04.tiff"
    result = reduced_datacube.save_result(format="GTiff")
    job = result.create_job()
    job.start_and_wait()
    job.get_results().download_files(target=exported_path)