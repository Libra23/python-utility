#!/usr/bin/env python3

import control.matlab as control
import numpy as np

class model_creator:
    def create_model_1st(self, K, T):
        self.P = control.tf([K], [T, 1])
        print('P = ', self.P)

    def create_model_2nd(self, K, wn, zeta):
        self.P = control.tf([wn**2 * K], [1, 2* zeta * wn, wn**2])
        print('P = ', self.P)

    def generate_dataset(self, u, dt, csv=False, plot=False):
        N = len(u)
        t = np.arange(0, N * dt, dt)
        y, _, _ = control.lsim(self.P, u, t)
        if csv:
            import pandas as pd
            df = pd.DataFrame({'time':t, 'input':u, 'output':y}).to_csv('sample.csv')
        if plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(t, u, label = "u")
            plt.plot(t, y, label = "y")
            plt.figure()
            control.bode(self.P)
            plt.show()
        return y, t

if __name__=="__main__":
    mc = model_creator()
    mc.create_model_1st(1.02, 0.74)
    u = np.ones(1000)
    dt = 0.001
    y, t = mc.generate_dataset(u, dt, True, True)