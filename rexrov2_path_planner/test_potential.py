from rexrov2_path_planner.potential_field import PotentialField

print("potential_field_planning start")

sx = 0.0  # start x position [m]
sy = 10.0  # start y positon [m]
gx = 30.0  # goal x position [m]
gy = 30.0  # goal y position [m]
ox = [15.0, 5.0, 20.0, 25.0]  # obstacle x position list [m]
oy = [25.0, 15.0, 26.0, 25.0]  # obstacle y position list [m]

reso = 0.5
rr = 10.0

poten_field = PotentialField()

path = poten_field.potential_field_planning(sx, sy, gx, gy, ox, oy, reso, rr)
print("----------- path -------------")
print(path)

pmap = poten_field.pmap
print("----------- pmap --------------")
print(pmap)
