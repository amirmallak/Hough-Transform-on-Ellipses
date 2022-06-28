import cv2 as cv
import numpy as np
import utilities as util
import matplotlib.pyplot as plt


def pre_processing_ellipse(im_color, sobel_edge_th, run_canny=None, run_keep_outside_shell=True):
    im = im_color
    im_resize_color = util.image_resize(im, (256, 256))
    im = util.image_gray(im_resize_color)
    im = util.image_blurring(im)
    if run_canny:
        im = cv.Canny(im.astype(np.uint8), run_canny[0], run_canny[1])
    edges, grads, grad_map = util.edge_gradient_orientation(im, sobel_edge_th)

    # Deleting the gradients on edges of the image (was created due to "cliff" of image edges)
    edges[0, :], edges[255, :], edges[:, 0], edges[:, 255] = [False] * 4

    # Keep only the outside shell of the ellipse edge map gradients
    if run_keep_outside_shell:
        new_edges, out_side_shell_grads = util.keep_outside_shell_generic(edges, grads)
        edges, grads = new_edges, out_side_shell_grads

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('original')
    plt.imshow(im_color, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title('edge magnitude')
    plt.imshow(edges, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title('edge direction')
    plt.imshow(grad_map, cmap='gray')

    return edges, grads, im_resize_color
