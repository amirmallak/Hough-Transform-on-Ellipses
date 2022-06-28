import numpy as np

from typing import Tuple, List


# class CandPoint:
#     x = None  # X coordinate point of P or Q (on ellipse perimeter)
#     y = None  # Y coordinate point of P or Q (on ellipse perimeter)
#     tangent = None  # The tangent line of P or Q
#     grad = None  # The gradient value of P or Q
#
#     def __init__(self, x, y, tangent, grad):
#         self.x = x
#         self.y = y
#         self.tangent = tangent
#         self.grad = grad


class VPoint:
    x = None  # X coordinate point on the TM line
    y = None  # Y coordinate point on the TM
    tm_line = None  # The corresponding line on witch the center ellipse point relies on
    # p: CandPoint = None  # Random point on the ellipse perimeter
    # q: CandPoint = None  # Random point on the ellipse perimeter
    p: Tuple = None  # Random point on the ellipse perimeter
    q: Tuple = None  # Random point on the ellipse perimeter

    def __init__(self, x, y, tm_line, p, q):
        self.x = int(x)
        # self.y = int(y)
        self.y = int(np.rint(y))
        self.tm_line = tm_line
        self.p = p
        self.q = q


class Grid:
    _cells = None
    _image_size = None
    _grid = None

    def __init__(self, image_size=256, cells=4):
        self._cells = cells
        self._image_size = image_size
        self._grid = np.empty((self._cells, self._cells), dtype=object)  # 2D array where each cell is a list of VPoint
        self._init_grid_()

    def _init_grid_(self):
        for i in range(self._grid.shape[0]):
            for j in range(self._grid.shape[1]):
                self._grid[i][j] = []

    def vote(self, v_point: VPoint):
        # steps = int((self._image_size / self._cells)) + 1
        steps = np.ceil(self._image_size / self._cells)
        x_cell = int(v_point.x / steps)
        y_cell = int(v_point.y / steps)

        self._grid[y_cell, x_cell].append(v_point)

    def get_max_voting(self):
        max = -1
        cor = []
        for ii in range(self._grid.shape[0]):
            for jj in range(self._grid.shape[1]):
                num_votes = len(self._grid[ii, jj])
                print(f'Cell: [{ii}, {jj}] got {num_votes} votes')
                if num_votes == max:
                    cor.append((ii, jj))
                if num_votes > max:
                    cor = []
                    cor.append([ii, jj])
                    max = num_votes

        print(cor)
        # if len(cor) != 0:
        #     p_list = self._grid[cor[0][0], cor[0][1]]
        #     return p_list[len(p_list) -1 ]

        if len(cor) != 0:
            v_list = cor[0]
            # v_mean = np.mean()
            p_list = self._grid[cor[0][0], cor[0][1]]
            xm = np.mean([p.x for p in p_list])
            ym = np.mean([p.y for p in p_list])
            # return p_list[len(p_list) - 1]

            return xm, ym

    def get_max_voting_2(self, threshold: int = 0):
        max_voting = -1
        cell = []
        num_votes = []
        for ii in range(self._grid.shape[0]):
            for jj in range(self._grid.shape[1]):
                num_votes.append(((ii, jj), len(self._grid[ii, jj])))
                # print(f'Cell: [{ii}, {jj}] got {num_votes} votes')

                # if num_votes[-1][1] > max_voting:
                #     cell = [ii, jj]
                #     max_voting = num_votes[-1][1]

        # print(f'The cell with the maximum voting was: {cell}, out of [{self._grid.shape[0]}, {self._grid.shape[1]}] '
        #       f'cells')

        # Sorting the num_votes by the number of votes
        num_votes.sort(key=lambda x: x[1])

        p_list = [self._grid[x[0][0], x[0][1]] for x in num_votes[-4:]]
        # p_list = self._grid[cell[0], cell[1]]
        # xm = np.mean([np.mean(z.x for z in p) for p in p_list])
        # ym = np.mean([np.mean(z.y for z in p) for p in p_list])
        points_list = [p[0] for p in p_list]
        xm = np.mean([p.x for p in points_list])
        ym = np.mean([p.y for p in points_list])

        # ------------------------------------------
        max_vote_cell = num_votes[-1][0]
        x_cell = max_vote_cell[0]
        y_cell = max_vote_cell[1]

        return xm, ym, x_cell, y_cell

    def get_threshold_voting(self, threshold: int = 0) -> List[Tuple]:
        max_voting = -1
        cell = []
        num_votes = []
        for ii in range(self._grid.shape[0]):
            for jj in range(self._grid.shape[1]):
                num_votes.append(((ii, jj), len(self._grid[ii, jj])))
                # print(f'Cell: [{ii}, {jj}] got {num_votes} votes')

                # if num_votes[-1][1] > max_voting:
                #     cell = [ii, jj]
                #     max_voting = num_votes[-1][1]

        # print(f'The cell with the maximum voting was: {cell}, out of [{self._grid.shape[0]}, {self._grid.shape[1]}] '
        #       f'cells')

        # Sorting the num_votes by the number of votes
        num_votes.sort(key=lambda x: x[1])

        # p_list = [self._grid[x[0][0], x[0][1]] for x in num_votes[-4:]]
        # p_list = self._grid[cell[0], cell[1]]
        # xm = np.mean([np.mean(z.x for z in p) for p in p_list])
        # ym = np.mean([np.mean(z.y for z in p) for p in p_list])
        # points_list = [p[0] for p in p_list]
        # xm = np.mean([p.x for p in points_list])
        # ym = np.mean([p.y for p in points_list])
        # center_points = [(x[0], x) for x in num_votes]
        center_points_threshold = [x for x in num_votes if x[1] > threshold]
        center_vpoints = [self._grid[x[0][0], x[0][1]] for x in center_points_threshold]
        center_points = [(p[0].x, p[0].y) for p in center_vpoints]

        return center_points
