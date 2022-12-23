import numpy as np
from collections import deque


class PotentialField:
    def __init__(self):
        self.kp = 5.0
        self.eta = 100.0
        self.oscillation_length = 3
        self.area_width = 30.0
        self.pmap = []
        self.minx = 0.0
        self.miny = 0.0

        self.motion = [[1, 0],
                       [0, 1],
                       [-1, 0],
                       [0, -1],
                       [-1, -1],
                       [-1, 1],
                       [1, -1],
                       [1, 1]]

    def att_potential(self, x, y, gx, gy):
        """
        Calculate the attractive potential at the given point based on the distance to the goal location.

        Parameters:
        - x, y (float): coordinates of the point
        - gx, gy (float): coordinates of the goal location

        Returns:
        float: attractive potential at the point
        """

        # attractive potential is proportional to the distance to the goal location
        return 0.5 * self.kp * np.hypot(x-gx, y-gy)

    def rep_potential(self, x, y, ox, oy, rr):
        """
        Calculate the repulsive potential at the given point based on the distance to the nearest obstacle.

        Parameters:
        - x, y (float): coordinates of the point
        - ox, oy (list): coordinates of the obstacles
        - rr (float): potential range of the obstacles

        Returns:
        float: repulsive potential at the point
        """

        # initialize the nearest obstacle and distance to infinity
        min_obstacle = None
        min_distance = float("inf")

        # find the nearest obstacle
        for ox, oy in zip(ox, oy):
            distance = np.hypot(x - ox, y - oy)  # distance to the obstacle
            if distance < min_distance:  # update the nearest obstacle if necessary
                min_distance = distance
                min_obstacle = (ox, oy)

        # calculate the repulsive potential
        if min_distance <= rr:  # inside the potential range
            if min_distance <= 0.1:  # avoid division by zero
                min_distance = 0.1
            # repulsive potential is proportional to the inverse of the distance to the nearest obstacle, with a stronger potential at closer distances
            return 0.5 * self.eta * (1.0 / min_distance - 1.0 / rr) ** 2
        else:  # outside the potential range
            return 0.0

    def oscillations_detection(self, previous_ids, ix, iy):
        """
        Check if the robot is getting stuck in an oscillation by checking for repeated cells in the list of previous cells visited.

        Parameters:
        - previous_ids (deque): list of previous cells visited by the robot
        - ix, iy (int): indices of the current cell

        Returns:
        bool: True if an oscillation is detected, False otherwise
        """
        previous_ids.append(
            (ix, iy))  # add the current cell to the list of previous cells

        # keep the list of previous cells at a fixed length by removing the oldest element
        if len(previous_ids) > self.oscillation_length:
            previous_ids.popleft()

        previous_ids_set = set()
        for i in previous_ids:
            if i in previous_ids_set:
                return True

            else:
                previous_ids_set.add(i)

        return False

    def cal_potential_field(self, gx, gy, ox, oy, reso, rr, sx, sy):
        """
        Calculate the potential field over a given area based on the locations of the goal, obstacles, and start point.

        Parameters:
        - gx, gy (float): coordinates of the goal location
        - ox, oy (list): coordinates of the obstacles
        - reso (float): resolution of the potential field (cell size)
        - rr (float): potential range of the obstacles
        - sx, sy (float): coordinates of the start point

        Returns:
        tuple: potential field, minimum x coordinate, minimum y coordinate
        """
        # determine the bounds of the potential field
        minx = min(min(ox), sx, gx) - self.area_width / 2.0
        miny = min(min(oy), sy, gy) - self.area_width / 2.0
        maxx = max(max(ox), sx, gx) + self.area_width / 2.0
        maxy = max(max(oy), sy, gy) + self.area_width / 2.0

        # calculate the size of the potential field in cells
        xw = int(round((maxx - minx) / reso))
        yw = int(round((maxy - miny) / reso))

        # initialize the potential field with zeros
        pmap = [[0.0 for i in range(yw)] for i in range(xw)]

        # calculate the potential at each cell
        for ix in range(xw):
            x = ix * reso + minx

            for iy in range(yw):
                y = iy * reso + miny
                ug = self.att_potential(
                    x, y, gx, gy)  # attractive potential
                uo = self.rep_potential(
                    x, y, ox, oy, rr)  # repulsive potential
                uf = ug + uo  # total potential
                pmap[ix][iy] = uf

        return pmap, minx, miny

    def potential_field_planning(self, sx, sy, gx, gy, ox, oy, reso, rr):
        """
        Find a path from the start point to the goal using the potential field.

        Parameters:
        - sx, sy (float): coordinates of the start point
        - gx, gy (float): coordinates of the goal location
        - ox, oy (list): coordinates of the obstacles
        - reso (float): resolution of the potential field (cell size)
        - rr (float): potential range of the obstacles

        Returns:
        list: list of coordinates for the path from the start point to the goal
        """

        # calculate the potential field
        self.pmap, self.minx, self.miny = self.cal_potential_field(
            gx, gy, ox, oy, reso, rr, sx, sy)
        # convert the start point and goal to cell indices
        ix = round((sx - self.minx) / reso)
        iy = round((sy - self.miny) / reso)
        gix = round((gx - self.minx) / reso)
        giy = round((gy - self.miny) / reso)

        # initialize the path with the start point
        path = [(ix, iy)]
        # initialize the list of previous cell indices
        previous_ids = deque()
        # initialize the flag for oscillation detection
        oscillation = False

        while True:
            # check if we have reached the goal
            if ix == gix and iy == giy:
                break

            # check if we are stuck in an oscillation
            oscillation = self.oscillations_detection(previous_ids, ix, iy)
            if oscillation:
                # backtrack to the last cell with a lower potential
                ix, iy = previous_ids[-2]
                path.pop()  # remove the oscillating point from the path
                previous_ids.clear()  # clear the list of previous cells
                continue

            minp = float("inf")
            minix, miniy = -1, -1
            # search the lowest potential around the current cell
            for i in range(len(self.motion)):
                inx = int(ix + self.motion[i][0])
                iny = int(iy + self.motion[i][1])
                if inx >= len(self.pmap) or iny >= len(self.pmap[0]) or inx < 0 or iny < 0:
                    continue
                if self.pmap[inx][iny] < minp:
                    minp = self.pmap[inx][iny]
                    minix = inx
                    miniy = iny

            # move to the cell with the lowest potential
            ix, iy = minix, miniy
            path.append((ix, iy))

        return path
