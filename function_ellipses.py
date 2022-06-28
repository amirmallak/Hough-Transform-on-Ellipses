from ellipse_utilities import *


def ellipse_images(image_name: str):
    images_names = {'two_tires': two_tires,
                    'three_ellipse': three_elipse,
                    'one_ellipse': one_elipse,
                    'watter_cup': watter_cup,
                    'box_image': box_image,
                    'gtest_image': gtest_image,
                    'headline_pic_bicycle': headline_pic_bicycle,
                    'gettyimages_roundabout': gettyimages_roundabout,
                    'wall_plates_image': wall_plates_image,
                    'very_long_truck_trailer':
                        very_long_truck_trailer_for_exceptional_transport_with_many_sturdy_tires,
                    'brondby_haveby_allotment':
                        brondby_haveby_allotment_gardens_copenhagen_denmark}

    images_names[image_name]()
