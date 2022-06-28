import cv2.cv2
import numpy as np

from grid import VPoint
from typing import Tuple, List
from scipy.signal import convolve2d

IMAGE_SHOW = True


def image_read(path: str) -> np.ndarray:
    return cv2.imread(path)


def show_image(name: str, image: np.ndarray):
    if not IMAGE_SHOW:
        return
    cv2.imshow(name, image)
    cv2.waitKey(0)


def image_gray(image: np.ndarray):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def image_resize(image: np.ndarray, new_dim: Tuple) -> np.ndarray:
    resized_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_CUBIC)

    return resized_image


def image_blurring(image: np.ndarray) -> np.ndarray:
    """
    A helper function

    Args:
        image: A numpy array representing the input image

    Returns: A blurred image (after applying a 2D convolution with a gaussian manually built mask)
    """

    gaussian_mask = (1 / 164) * np.array([[1, 8, 1],
                                         [8, 128, 8],
                                         [1, 8, 1]])

    image = convolve2d(image, gaussian_mask, mode='same')

    return image


def canny_edge_detector(image: np.ndarray, canny_th_low: int, canny_th_high: int) -> np.ndarray:

    image_canny = cv2.Canny(image, canny_th_low, canny_th_high)

    return image_canny


def edge_gradient_orientation(image: np.ndarray, threshold) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sobel_x_mask = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])
    sobel_y_mask = sobel_x_mask.T

    sobel_edge_x = convolve2d(image, sobel_x_mask, mode='same')
    sobel_edge_y = convolve2d(image, sobel_y_mask, mode='same')

    sobel_edge_magnitude = (sobel_edge_x * sobel_edge_x + sobel_edge_y * sobel_edge_y) ** 0.5
    sobel_edge_map = sobel_edge_magnitude > threshold

    # Calculating the gradient direction
    sobel_gradient_direction = np.arctan2(sobel_edge_y, sobel_edge_x) * (180 / np.pi)
    sobel_gradient_direction += 180
    sobel_gradient_direction *= sobel_edge_map

    # Coloring the different gradient orientations
    red = np.array([0, 0, 255])
    cyan = np.array([255, 255, 0])
    green = np.array([0, 255, 0])
    yellow = np.array([0, 255, 255])

    sobel_grad_orien = np.zeros((sobel_gradient_direction.shape[0], sobel_gradient_direction.shape[1], 3),
                                dtype=np.uint8)

    # setting the colors, maybe there is a better way, my numpy skills are rusty
    # it checks that magnitude is above the threshold and that the orientation is in range
    sobel_grad_orien[sobel_gradient_direction < 90] = red
    sobel_grad_orien[(sobel_gradient_direction > 90) & (sobel_gradient_direction < 180)] = cyan
    sobel_grad_orien[(sobel_gradient_direction > 180) & (sobel_gradient_direction < 270)] = green
    sobel_grad_orien[(sobel_gradient_direction > 270)] = yellow

    # Discarding the background gradient orientation
    sobel_grad_orien[:, :, 0] *= sobel_edge_map
    sobel_grad_orien[:, :, 1] *= sobel_edge_map
    sobel_grad_orien[:, :, 2] *= sobel_edge_map

    return sobel_edge_map, sobel_gradient_direction,  sobel_grad_orien


def calc_tangent(grad: float) -> float:

    tangent = grad - 90

    if grad > 180:
        tangent -= 180

    tangent *= -1

    return tangent


def calc_TM(p, q, eps1, eps2):
    # global a, b
    x1, y1 = p
    x2, y2 = q

    t1 = (y1-y2 - (x1*eps1) + (x2*eps2)) / (eps2-eps1 + 1e-3)
    t2 = (eps1*eps2*(x2-x1) - y2*eps1 + y1*eps2) / (eps2-eps1 + 1e-3)

    m1 = (x1+x2) / 2
    m2 = (y1+y2) / 2

    a = (t2 - m2) / (t1 - m1 + 1e-3)
    b = (m2 * t1 - m1 * t2) / (t1 - m1 + 1e-3)

    def y(x):

        return a * x + b

    return y, a, b


def keep_outside_shell(edge_map: np.ndarray, gradient_direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Arrays of ones as size of edge map
    left_to_right = np.ones(edge_map.shape)
    right_to_left = np.ones(edge_map.shape)

    # Loop through the rows from left to right
    for i in range(edge_map.shape[0]):
        for j in range(edge_map.shape[1]):
            if edge_map[i, j]:
                left_to_right[i, j] = 0
                break

    # Loop through the rows from right to left
    for i in range(edge_map.shape[0]):
        for j in range((edge_map.shape[1] - 1), -1, -1):
            if edge_map[i, j]:
                right_to_left[i, j] = 0
                break

    outside_shell = left_to_right * right_to_left
    outside_shell = (outside_shell + 1) * (outside_shell == 0)

    outside_shell_grad = gradient_direction * outside_shell
    new_edge_map = outside_shell == 1

    return new_edge_map, outside_shell_grad


def keep_outside_shell_generic(edge_map: np.ndarray, gradient_direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    indices_array = np.argwhere(edge_map)
    new_edge_map = edge_map.copy()
    outside_shell_grad = gradient_direction.copy()
    percentage = 3e-1

    radius = int(edge_map.shape[0] * 5e-2)

    # Detect the inner ellipse cells
    for cell in indices_array:
        r = cell[0]
        c = cell[1]

        if (r < radius) or (r > gradient_direction.shape[0] - radius) or \
           (c < radius) or (c > gradient_direction.shape[1] - radius):
            continue

        cell_val = gradient_direction[r, c]

        mask_right: np.ndarray = gradient_direction[r - radius:r + radius, c + 1:(c + radius+1)]
        non_zero_indices: Tuple = np.nonzero(mask_right)
        non_zero_elements_right: np.ndarray = mask_right[non_zero_indices]

        mask_left: np.ndarray = gradient_direction[r - radius:r + radius, c - radius-1:c - 1]
        non_zero_indices: Tuple = np.nonzero(mask_left)
        non_zero_elements_left: np.ndarray = mask_left[non_zero_indices]

        # Sort in place
        non_zero_elements_right.sort()
        non_zero_elements_left.sort()

        criteria_right = int(non_zero_elements_right.shape[0] * percentage)
        criteria_left = int(non_zero_elements_left.shape[0] * percentage)

        # Max percentage elements are bigger than current
        max_percentage_elements_right = non_zero_elements_right[-criteria_right:]
        bool_increase_right = (cell_val < max_percentage_elements_right).all()

        max_percentage_elements_left = non_zero_elements_left[-criteria_left:]
        bool_increase_left = (cell_val > max_percentage_elements_left).all()

        # Min percentage elements are smaller than current
        min_percentage_elements_right = non_zero_elements_right[:criteria_right]
        bool_decrease_right = (cell_val > min_percentage_elements_right).all()

        min_percentage_elements_left = non_zero_elements_left[:criteria_left]
        bool_decrease_left = (cell_val < min_percentage_elements_left).all()

        right_criteria = ((0 <= cell_val <= 180) and (not bool_decrease_right)) or \
                         ((180 < cell_val <= 360) and (not bool_increase_right))
        left_criteria = ((0 <= cell_val <= 180) and (not bool_decrease_left)) or \
                        ((180 < cell_val <= 360) and (not bool_increase_left))

        if right_criteria:
            new_edge_map[r, c] = False

    outside_shell_grad *= new_edge_map

    return new_edge_map, outside_shell_grad


def choose_ellipse_coor(edges: np.ndarray, gradient_direction: np.ndarray) -> List[Tuple]:
    cand_coor = np.argwhere(edges)

    cand_coor: List[List] = [list(x) for x in cand_coor]
    dist = np.inf
    above_threshold = 8e1
    beneath_threshold = 5e0
    point_pair = []
    remove_list = []

    for p_ind, p in enumerate(cand_coor):
        criteria = False
        flag = False
        q_ind = p_ind
        if q_ind == len(cand_coor):
            break
        while not criteria and q_ind != len(cand_coor) - 1:  # Continue searching for the next valid point
            q_ind += 1
            q = cand_coor[q_ind]
            if p in remove_list or q in remove_list:
                flag = True
                break
            dist = np.sqrt(np.power((q[1] - p[1]), 2) + np.power((q[0] - p[0]), 2))

            criteria = (dist < above_threshold) and (gradient_direction[p[0], p[1]] - gradient_direction[q[0], q[1]] < 1e2)

        if not flag:
            point_pair.append((p, q))
            remove_list.append(q)

    return point_pair


def choose_ellipse_coor_2(edges: np.ndarray, gradient_direction: np.ndarray) -> List[Tuple]:
    cand_coor = np.argwhere(edges)

    cand_coor: List[List] = [list(x) for x in cand_coor]

    # above_threshold = 1e2
    beneath_threshold = 5e0
    tolerance = 1e1
    point_pair = []
    for r in range(0, len(cand_coor), 50):
        for l in range(len(cand_coor) - 1, -1, -50):
            p = cand_coor[r]
            q = cand_coor[l]

            dist = np.sqrt(np.power((q[1] - p[1]), 2) + np.power((q[0] - p[0]), 2))
            grad_diff = gradient_direction[p[0], p[1]] - gradient_direction[q[0], q[1]]

            criteria_dist = abs(p[1] - q[1]) > 3 and abs(p[0] - q[0] > 3) and r != l
            if criteria_dist:
                point_pair.append((p, q))

    return point_pair


def get_bdc(grid, x_cell, y_cell):
    ellipse_edge_points: List[VPoint] = grid._grid[x_cell, y_cell]

    b_values = np.linspace(-1e2, 1e2, 10)
    d_values = np.linspace(-1e1, 1e1, 10)

    vote = np.empty((len(b_values), len(d_values), 1e4), dtype=object)
    for i in vote.shape[0]:
        for j in vote.shape[1]:
            for k in vote.shape[2]:
                vote[i, j, k] = [0, []]

    for v_point in ellipse_edge_points:
        for b_ind, b in enumerate(b_values):
            for d_ind, d in enumerate(d_values):
                if np.power(d, 2) >= abs(b):
                    continue

                # Calculate c param for each point - p and q
                x_p = v_point.p[0]
                y_p = v_point.p[1]
                c_p = -(x_p**2 + b * y_p**2 + 2 * d * x_p * y_p)

                x_q = v_point.q[0]
                y_q = v_point.q[1]
                c_q = -(x_q ** 2 + b * y_q ** 2 + 2 * d * x_q * y_q)

                v_list = vote[b_ind, d_ind, np.rint(c_p / 1e3)]
                v_list[0] += 1
                v_list[1].append(c_p)
                vote[b_ind, d_ind, np.rint(c_p / 1e3)] = v_list

                v_list = vote[b_ind, d_ind, np.rint(c_q / 1e3)]
                v_list[0] += 1
                v_list[1].append(c_q)
                vote[b_ind, d_ind, np.rint(c_q / 1e3)] = v_list

    max = 0
    for b in vote.shape[0]:
        for d in vote.shape[1]:
            for c in vote.shape[2]:
                cell = vote[b, d, c]

                if cell[0] > max:
                    cell_ind = (b, d, c)
                    max = cell[0]

    b = cell_ind[0]
    d = cell_ind[1]
    c = np.mean([x for x in vote[cell_ind][1]])

    return b, d, c
