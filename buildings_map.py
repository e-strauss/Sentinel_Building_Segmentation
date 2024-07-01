from pyrosm import get_data, OSM


def create_buildings_map(city, n=1):
    fp = get_data(city)
    osm = OSM(fp)
    buildings = osm.get_buildings()
    ax = buildings.plot(figsize=(10 * n, 10 * n))
    ax.figure.savefig("data/map-{}.pdf".format(city))
    bounds = buildings.total_bounds
    return bounds
