# This is a sample Python script.
from data_acquisition_and_alignment import data_acquisition_and_alignment_pipeline, data_alignment_and_save_pipeline, \
    get_test_data_of_berlin
from data_preparation import merge_patches, make_batches, create_dataset_for_building_segmentation

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Hello World!")
    cities = ['Dresden', 'Leipzig', 'Hamburg', 'Hannover', 'Muenchen', 'Paris', 'London', 'Copenhagen',
              'Erfurt']
    # data_acquisition_and_alignment_pipeline(cities[1:2])
    # data_alignment_and_save_pipeline(cities)
    # make_batches(cities)
    i = 0
    # create_dataset_for_building_segmentation(cities[i:i+1])
    create_dataset_for_building_segmentation(cities)
    # get_test_data_of_berlin()
