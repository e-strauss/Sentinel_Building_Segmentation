# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from pyrosm import get_data, OSM
import matplotlib.pyplot as plt


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def get_city_osm(city_name):
    fp = get_data(city_name)
    osm = OSM(fp)
    buildings = osm.get_data_by_custom_criteria(custom_filter={'building': True})
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    bs = len(buildings.geometry)
    print(bs)
    bs = bs // 1000
    p = 0.1
    # Plotting buildings
    for i, polygon in enumerate(buildings.geometry):
        if i % bs == 0:
            print(p)
            p += 0.1
            if p>10:
                break
        if polygon.geom_type == 'Polygon':
            x, y = polygon.exterior.xy
            ax.fill(x, y, alpha=0.7, fc='blue', ec='none')
        elif polygon.geom_type == 'MultiPolygon':
            continue
            for subpolygon in polygon:
                x, y = subpolygon.exterior.xy
                ax.fill(x, y, alpha=0.7, fc='blue', ec='none')

    # Setting plot title
    plt.title("Buildings from OpenStreetMap - Berlin")
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    get_city_osm("Berlin")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
