from rexrov2_path_planner.potential_field import PotentialField
import matplotlib.pyplot as plt
import numpy as np

print("potential_field_planning start")

sx = 5.0  # start x position [m]
sy = 0.0  # start y positon [m]
gx = 30.0  # goal x position [m]
gy = 0.0  # goal y position [m]
ox = [-1.0]  # obstacle x position list [m]
oy = [-1.0]  # obstacle y position list [m]

reso = 0.5
rr = 10.0

poten_field = PotentialField()

path = poten_field.potential_field_planning(sx, sy, gx, gy, ox, oy, reso, rr)
print("----------- path -------------")
print(len(path))
path_array = np.array(path)
print(path_array)

plt.plot(path_array[:, 0], path_array[:, 1])
s = [10 * (rr / reso)] * len(ox)
plt.scatter(ox, oy, s=s, c="red")
plt.show()

pmap = poten_field.pmap
# print("----------- pmap --------------")
# print(pmap)
