# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import buildings_map


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    cities = ['dresden', 'Berlin','leipzig', 'hamburg', 'Hannover', 'Muenchen','Paris', 'London']
    with open("data/city_coords.txt", 'w') as f:
        for city in cities:
            coords = buildings_map.create_buildings_map(city)
            f.write(city + " " + str(coords) + "\n")
            print(coords)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
