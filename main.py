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
    cities = ['dresden', 'leipzig', 'hamburg', 'Frankfurt am Main', 'Hannover', 'Berlin', 'Muenchen','Paris', 'London']
    for city in cities[2:4]:
        coords = buildings_map.create_buildings_map(city)
        print(coords)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
