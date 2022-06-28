import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from find_ellipse import find_elipse
from preprocessing import pre_processing_ellipse

img_dir_path = './ellipses'
res_dir_path = "./results"
RES_IM_NAME = ''


def save_final():
    plt.savefig(os.path.join(res_dir_path, '{}_final.jpg'.format(RES_IM_NAME)))


def save_progress():
    plt.savefig(os.path.join(res_dir_path, '{}_progress.jpg'.format(RES_IM_NAME)))


def one_elipse():
    global RES_IM_NAME
    RES_IM_NAME = "one_elipse"

    im_file_name = ".//ellipses//test.png"
    im_path = im_file_name
    im_color = cv.imread(im_path)

    Sobel_edge_th = 500
    cells_grid_count = 20
    voting_th = 150
    steps = 50
    item = np.array([[0, 255], [0, 255]])

    edges, grads, im_resize_color = pre_processing_ellipse(im_color, Sobel_edge_th)
    im_resize_color, center_pts, axis = find_elipse(cells_grid_count, voting_th, item, steps, edges, grads,
                                                    im_resize_color)

    plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)
    plt.imshow(im_resize_color, cmap='gray')

    save_progress()
    plt.show()

    image_ellipse = cv.ellipse(im_resize_color, (int(center_pts[0]), int(center_pts[1])), axis, 0, 0, 360, (0, 0, 255),
                               1)

    plt.figure()
    plt.title('Ellipse')
    plt.imshow(image_ellipse)
    plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)

    save_final()
    plt.show()


def three_elipse():
    global RES_IM_NAME
    RES_IM_NAME = "three_elipse"

    im_file_name = ".//ellipses//test3.png"
    im_path = im_file_name
    im_color = cv.imread(im_path)

    Sobel_edge_th = 300
    cells_grid_count = 20
    voting_th = 150
    steps = 30

    _hyperparameter = np.array([
        [[0, 150], [150, 255]],
        [[0, 150], [0, 120]],
        [[150, 255], [100, 200]]
    ])

    center_pts_list = []
    axis_list = []

    edges, grads, im_resize_color = pre_processing_ellipse(im_color, Sobel_edge_th)
    for item in _hyperparameter:
        im_resize_color, center_pts, axis = find_elipse(cells_grid_count, voting_th, item, steps, edges, grads,
                                                        im_resize_color)
        center_pts_list.append(center_pts)
        axis_list.append(axis)

    for center_pts, axis in zip(center_pts_list, axis_list):
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)
        plt.imshow(im_resize_color, cmap='gray')

    save_progress()
    plt.show()

    plt.figure()
    plt.title('Ellipse')

    image_ellipse = im_resize_color
    for center_pts, axis in zip(center_pts_list, axis_list):
        image_ellipse = cv.ellipse(image_ellipse, (int(center_pts[0]), int(center_pts[1])), axis, 0, 0, 360,
                                   (0, 0, 255), 1)
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)

    plt.imshow(image_ellipse)

    save_final()
    plt.show()


def two_tires():
    global RES_IM_NAME
    RES_IM_NAME = 'two_tires'

    im_file_name = "ellipses//nEKGD2wNiwqrTOc63kiWZT7b4.png"
    im_path = im_file_name
    im_color = cv.imread(im_path)

    Sobel_edge_th = 200
    cells_grid_count = 20
    voting_th = 150
    steps = 50

    _hyperparameter = np.array([
        [[50, 110], [75, 255]],
        [[115, 200], [50, 255]]
    ])

    center_pts_list = []
    axis_list = []

    edges, grads, im_resize_color = pre_processing_ellipse(im_color, Sobel_edge_th)
    for item in _hyperparameter:
        im_resize_color, center_pts, axis = find_elipse(cells_grid_count, voting_th, item, steps, edges, grads,
                                                        im_resize_color)
        center_pts_list.append(center_pts)
        axis_list.append(axis)

    for center_pts, axis in zip(center_pts_list, axis_list):
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)
        plt.imshow(im_resize_color, cmap='gray')

    save_progress()
    plt.show()

    plt.figure()
    plt.title('Ellipse')

    image_ellipse = im_resize_color
    for center_pts, axis in zip(center_pts_list, axis_list):
        image_ellipse = cv.ellipse(image_ellipse, (int(center_pts[0]), int(center_pts[1])), axis, 0, 0, 360,
                                   (0, 0, 255), 1)
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)

    plt.imshow(image_ellipse)

    save_final()

    plt.show()


def box_image():
    global RES_IM_NAME
    RES_IM_NAME = 'box_image'

    im_file_name = "ellipses//images.jpg"
    im_path = im_file_name
    im_color = cv.imread(im_path)

    canny_th1 = 150
    canny_th2 = 700
    Sobel_edge_th = 1010
    cells_grid_count = 20
    voting_th = 150
    steps = 10

    _hyperparameter = np.array([
        [[63, 119], [114, 206]],
        [[121, 176], [117, 195]],
        [[118, 188], [18, 105]],
    ])

    center_pts_list = []
    axis_list = []

    edges, grads, im_resize_color = pre_processing_ellipse(im_color, Sobel_edge_th, [canny_th1, canny_th2], True)
    for item in _hyperparameter:
        im_resize_color, center_pts, axis = find_elipse(cells_grid_count, voting_th, item, steps, edges, grads,
                                                        im_resize_color)
        center_pts_list.append(center_pts)
        axis_list.append(axis)

    for center_pts, axis in zip(center_pts_list, axis_list):
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)
        plt.imshow(im_resize_color, cmap='gray')

    save_progress()
    plt.show()

    plt.figure()
    plt.title('Ellipse')

    image_ellipse = im_resize_color
    for center_pts, axis in zip(center_pts_list, axis_list):
        image_ellipse = cv.ellipse(image_ellipse, (int(center_pts[0]), int(center_pts[1])), axis, 0, 0, 360,
                                   (0, 0, 255), 1)
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)

    plt.imshow(image_ellipse)

    save_final()
    plt.show()


def headline_pic_bicycle():
    global RES_IM_NAME
    RES_IM_NAME = 'headline_pic_bicycle'

    im_file_name = "ellipses//Headline-Pic.jpg"
    im_path = im_file_name
    im_color = cv.imread(im_path)

    Sobel_edge_th = 300
    cells_grid_count = 20
    voting_th = 150
    steps = 30

    _hyperparameter = np.array([
        [[80, 117], [160, 235]],
        [[130, 177], [167, 250]],
        [[10, 31], [152, 210]],
        [[37, 63], [157, 220]]
    ])

    center_pts_list = []
    axis_list = []

    edges, grads, im_resize_color = pre_processing_ellipse(im_color, Sobel_edge_th, None, True)

    for item in _hyperparameter:
        im_resize_color, center_pts, axis = find_elipse(cells_grid_count, voting_th, item, steps, edges, grads,
                                                        im_resize_color)
        center_pts_list.append(center_pts)
        axis_list.append(axis)

    for center_pts, axis in zip(center_pts_list, axis_list):
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)
        plt.imshow(im_resize_color, cmap='gray')

    save_progress()
    plt.show()

    plt.figure()
    plt.title('Ellipse')

    image_ellipse = im_resize_color
    for center_pts, axis in zip(center_pts_list, axis_list):
        image_ellipse = cv.ellipse(image_ellipse, (int(center_pts[0]), int(center_pts[1])), axis, 0, 0, 360,
                                   (0, 0, 255), 1)
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)

    plt.imshow(image_ellipse)
    save_final()
    plt.show()


def very_long_truck_trailer_for_exceptional_transport_with_many_sturdy_tires():
    global RES_IM_NAME
    RES_IM_NAME = 'very_long_truck'

    im_file_name = "ellipses//72384675-very-long-truck-trailer-for-exceptional-transport-with-many-sturdy-tires.webp"
    im_path = im_file_name
    im_color = cv.imread(im_path)

    canny_th1 = 300
    canny_th2 = 500
    Sobel_edge_th = 100
    cells_grid_count = 20
    voting_th = 150
    steps = 50

    _hyperparameter = np.array([
        [[180, 240], [50, 210]],
        [[110, 165], [34, 206]],
        [[70, 107], [34, 158]],
        [[22, 67], [33, 133]],
    ])

    center_pts_list = []
    axis_list = []

    edges, grads, im_resize_color = pre_processing_ellipse(im_color, Sobel_edge_th, [canny_th1, canny_th2], True)

    for item in _hyperparameter:
        im_resize_color, center_pts, axis = find_elipse(cells_grid_count, voting_th, item, steps, edges, grads,
                                                        im_resize_color)
        center_pts_list.append(center_pts)
        axis_list.append(axis)

    for center_pts, axis in zip(center_pts_list, axis_list):
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)
        plt.imshow(im_resize_color, cmap='gray')

    save_progress()
    plt.show()

    plt.figure()
    plt.title('Ellipse')

    image_ellipse = im_resize_color
    for center_pts, axis in zip(center_pts_list, axis_list):
        image_ellipse = cv.ellipse(image_ellipse, (int(center_pts[0]), int(center_pts[1])), axis, 0, 0, 360,
                                   (0, 0, 255), 1)
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)

    plt.imshow(image_ellipse)

    save_final()
    plt.show()


def brondby_haveby_allotment_gardens_copenhagen_denmark():
    global RES_IM_NAME
    RES_IM_NAME = 'brondby_haveby_allotment'

    im_file_name = "ellipses//5da02f8f443e6-brondby-haveby-allotment-gardens-copenhagen-denmark-7.jpg.png"
    im_path = im_file_name
    im_color = cv.imread(im_path)

    canny_th1 = 200
    canny_th2 = 600
    Sobel_edge_th = 200
    cells_grid_count = 20
    voting_th = 150
    steps = 30

    _hyperparameter = np.array([
        [[130, 252], [151, 245]],
        [[2, 121], [139, 235]],
        [[11, 114], [66, 130]],
        [[22, 238], [172, 147]],
    ])

    center_pts_list = []
    axis_list = []

    edges, grads, im_resize_color = pre_processing_ellipse(im_color, Sobel_edge_th, [canny_th1, canny_th2], True)
    for item in _hyperparameter:
        im_resize_color, center_pts, axis = find_elipse(cells_grid_count, voting_th, item, steps, edges, grads,
                                                        im_resize_color)
        center_pts_list.append(center_pts)
        axis_list.append(axis)

    for center_pts, axis in zip(center_pts_list, axis_list):
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)
        plt.imshow(im_resize_color, cmap='gray')

    save_progress()
    plt.show()

    plt.figure()
    plt.title('Ellipse')

    image_ellipse = im_resize_color
    for center_pts, axis in zip(center_pts_list, axis_list):
        image_ellipse = cv.ellipse(image_ellipse, (int(center_pts[0]), int(center_pts[1])), axis, 0, 0, 360,
                                   (0, 0, 255), 1)
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)

    plt.imshow(image_ellipse)

    save_final()
    plt.show()


def wall_plates_image():
    global RES_IM_NAME
    RES_IM_NAME = 'wall_plates_image'

    im_file_name = "ellipses//1271488188_2077d21f46_b.jpg"
    im_path = im_file_name
    im_color = cv.imread(im_path)

    canny_th1 = 70
    canny_th2 = 100
    Sobel_edge_th = 1000
    cells_grid_count = 20
    voting_th = 150
    steps = 5

    _hyperparameter = np.array([
        [[148, 201], [144, 235]],
        [[100, 143], [139, 220]],
        [[68, 100], [141, 213]],
        [[148, 206], [53, 141]],
        [[104, 146], [68, 141]],
        [[70, 105], [8, 68]],
    ])
    center_pts_list = []
    axis_list = []

    edges, grads, im_resize_color = pre_processing_ellipse(im_color, Sobel_edge_th, [canny_th1, canny_th2], True)
    for item in _hyperparameter:
        im_resize_color, center_pts, axis = find_elipse(cells_grid_count, voting_th, item, steps, edges, grads,
                                                        im_resize_color)
        center_pts_list.append(center_pts)
        axis_list.append(axis)

    for center_pts, axis in zip(center_pts_list, axis_list):
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)
        plt.imshow(im_resize_color, cmap='gray')

    save_progress()
    plt.show()

    plt.figure()
    plt.title('Ellipse')

    image_ellipse = im_resize_color
    for center_pts, axis in zip(center_pts_list, axis_list):
        image_ellipse = cv.ellipse(image_ellipse, (int(center_pts[0]), int(center_pts[1])), axis, 0, 0, 360,
                                   (0, 0, 255), 1)
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)

    plt.imshow(image_ellipse)

    save_final()
    plt.show()


def gettyimages_roundabout():
    global RES_IM_NAME
    RES_IM_NAME = 'gettyimages_roundabout'

    im_file_name = "ellipses//gettyimages-1212455495-612x612.jpg"
    im_path = im_file_name
    im_color = cv.imread(im_path)

    canny_th1 = 150
    canny_th2 = 700
    Sobel_edge_th = 500
    cells_grid_count = 20
    voting_th = 150
    steps = 30

    _hyperparameter = np.array([
        [[60, 140], [160, 255]],
        [[110, 220], [45, 200]]
    ])

    center_pts_list = []
    axis_list = []

    edges, grads, im_resize_color = pre_processing_ellipse(im_color, Sobel_edge_th, [canny_th1, canny_th2], True)
    for item in _hyperparameter:
        im_resize_color, center_pts, axis = find_elipse(cells_grid_count, voting_th, item, steps, edges, grads,
                                                        im_resize_color)
        center_pts_list.append(center_pts)
        axis_list.append(axis)

    for center_pts, axis in zip(center_pts_list, axis_list):
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)
        plt.imshow(im_resize_color, cmap='gray')

    save_progress()
    plt.show()

    plt.figure()
    plt.title('Ellipse')

    image_ellipse = im_resize_color
    for center_pts, axis in zip(center_pts_list, axis_list):
        image_ellipse = cv.ellipse(image_ellipse, (int(center_pts[0]), int(center_pts[1])), axis, 0, 0, 360,
                                   (0, 0, 255), 1)
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)

    plt.imshow(image_ellipse)

    save_final()
    plt.show()


def watter_cup():
    global RES_IM_NAME
    RES_IM_NAME = 'watter_cup'

    im_file_name = "ellipses//s-l400.jpg"
    im_path = im_file_name
    im_color = cv.imread(im_path)

    canny_th1 = 200
    canny_th2 = 100
    Sobel_edge_th = 700
    cells_grid_count = 20
    voting_th = 150
    steps = 10

    _hyperparameter = np.array([
        [[77, 188], [137, 174]],
        [[72, 193], [99, 124]],
    ])

    center_pts_list = []
    axis_list = []

    edges, grads, im_resize_color = pre_processing_ellipse(im_color, Sobel_edge_th, [canny_th1, canny_th2], True)
    for item in _hyperparameter:
        im_resize_color, center_pts, axis = find_elipse(cells_grid_count, voting_th, item, steps, edges, grads,
                                                        im_resize_color)
        center_pts_list.append(center_pts)
        axis_list.append(axis)

    for center_pts, axis in zip(center_pts_list, axis_list):
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)
        plt.imshow(im_resize_color, cmap='gray')

    save_progress()
    plt.show()

    plt.figure()
    plt.title('Ellipse')

    image_ellipse = im_resize_color
    for center_pts, axis in zip(center_pts_list, axis_list):
        image_ellipse = cv.ellipse(image_ellipse, (int(center_pts[0]), int(center_pts[1])), axis, 0, 0, 360,
                                   (0, 0, 255), 1)
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)

    plt.imshow(image_ellipse)

    save_final()
    plt.show()


def gtest_image():
    global RES_IM_NAME
    RES_IM_NAME = 'gtest_image'

    im_file_name = "ellipses//gtest.png"
    im_path = im_file_name
    im_color = cv.imread(im_path)

    canny_th1 = 0
    canny_th2 = 0
    Sobel_edge_th = 300
    cells_grid_count = 20
    voting_th = 150
    steps = 10

    _hyperparameter = np.array([
        [[63, 119], [114, 206]],
        [[121, 176], [117, 195]],
        [[118, 188], [18, 105]],
    ])

    center_pts_list = []
    axis_list = []

    edges, grads, im_resize_color = pre_processing_ellipse(im_color, Sobel_edge_th, [canny_th1, canny_th2], True)
    for item in _hyperparameter:
        im_resize_color, center_pts, axis = find_elipse(cells_grid_count, voting_th, item, steps, edges, grads,
                                                        im_resize_color)
        center_pts_list.append(center_pts)
        axis_list.append(axis)

    for center_pts, axis in zip(center_pts_list, axis_list):
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)
        plt.imshow(im_resize_color, cmap='gray')

    save_progress()
    plt.show()

    plt.figure()
    plt.title('Ellipse')

    image_ellipse = im_resize_color
    for center_pts, axis in zip(center_pts_list, axis_list):
        image_ellipse = cv.ellipse(image_ellipse, (int(center_pts[0]), int(center_pts[1])), axis, 0, 0, 360,
                                   (0, 0, 255), 1)
        plt.scatter(center_pts[0], center_pts[1], marker="x", color="green", alpha=1, s=100)

    plt.imshow(image_ellipse)
    save_final()
    plt.show()
