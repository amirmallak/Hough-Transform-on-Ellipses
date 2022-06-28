import utilities as util
import matplotlib.pyplot as plt

from grid import *


def find_elipse(cells_grid_count, voting_th, hyperparam, steps, edges, grads, im_resize_color):

    edges_coor = np.argwhere(edges)
    grid = Grid(cells=cells_grid_count)

    all_x = []
    all_y = []

    # iterating trough image points
    for r in range(0, len(edges_coor), steps):
        for c in range(len(edges_coor) - 1, -1, -steps):

            if r == c:
                continue
            if abs(edges_coor[c][1] - edges_coor[r][1]) < 3 or abs(edges_coor[c][0] - edges_coor[r][0]) < 3:
                continue

            p = (edges_coor[r][1], edges_coor[r][0])
            q = (edges_coor[c][1], edges_coor[c][0])

            if not (((hyperparam[0, 0] < p[0] < hyperparam[0, 1]) and (hyperparam[1, 0] < p[1] < hyperparam[1, 1])) and
                    ((hyperparam[0, 0] < q[0] < hyperparam[0, 1]) and (hyperparam[1, 0] < q[1] < hyperparam[1, 1]))):
                continue

            all_x.append(p[0])
            all_y.append(p[1])
            all_x.append(q[0])
            all_y.append(q[1])

            grad_p = grads[p[1], p[0]]
            grad_q = grads[q[1], q[0]]

            tangent_p = util.calc_tangent(grad_p)
            tangent_q = util.calc_tangent(grad_q)

            # Calculate an actual slope (not degree)
            tangent_p = np.tan(tangent_p * np.pi / 180)
            tangent_q = np.tan(tangent_q * np.pi / 180)

            p_tm = (p[0], 255 - p[1])
            q_tm = (q[0], 255 - q[1])
            TM, a, b = util.calc_TM(p_tm, q_tm, tangent_p, tangent_q)

            x = np.linspace(0, 256, 256)
            y = a * x + b
            y = 255 - y

            for i in range(255):
                y_ = TM(i)
                y_ = 255 - y_
                if 0 <= y_ <= 255:
                    vp = VPoint(i, y_, TM, p_tm, q_tm)
                    grid.vote(vp)

            edge_pts = np.array([[p[0], p[1]], [q[0], q[1]]])

            plt.subplot(2, 2, 4)

            plt.plot(x, y, '-r', alpha=0.10, label=f'y = {a}x + {b}')

            plt.title('TMs and Ellipse Center')
            plt.scatter(edge_pts[:, 0], edge_pts[:, 1], marker="x", color="blue", s=100)
            plt.imshow(im_resize_color, cmap='gray')

    center = grid.get_max_voting_2(voting_th)

    center_pts = (center[0], center[1])

    mmx = int((abs(np.max(all_x) - np.min(all_x)) / 2))
    mmy = int((abs(np.max(all_y) - np.min(all_y)) / 2))

    axis = (mmx, mmy)

    return im_resize_color, center_pts, axis
