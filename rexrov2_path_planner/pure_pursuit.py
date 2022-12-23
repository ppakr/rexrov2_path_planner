import numpy as np


class PurePursuit():
    def __init__(self, kp=0.5):
        self.kp = kp  # speed propotional gain

    def pure_pursuit(self, target, interceptor):
        # calculate v_d
        p_e_n = interceptor - target
        v_d = - self.kp * (p_e_n)/np.linalg.norm(p_e_n)
        chi_d = np.arctan2(v_d[1], v_d[0])
        u_d = np.linalg.norm(v_d)

        return chi_d, u_d
