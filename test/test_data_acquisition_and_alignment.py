from unittest import TestCase

from data_acquisition_and_alignment import align_buildings_to_sentinel, plot_city, create_buildings_map, \
    create_sentinel_map


class Test(TestCase):
    def test_align_buildings_to_sentinel(self):
        city = "berlin"
        img, bld = align_buildings_to_sentinel(city)
        plot_city(city, img, bld, (3, 3))

    def test_pipeline(self):
        city = "Berlin"
        a = create_buildings_map(city)
        b = ["west", "south", "east", "north"]
        # coords = {bi: float(a[i]) for i, bi in enumerate(b)}
        # create_sentinel_map(city, coords)
        img, bld = align_buildings_to_sentinel(city)
        plot_city(city, img, bld)
