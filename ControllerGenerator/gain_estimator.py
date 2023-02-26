#!/usr/bin/env python3

import control.matlab as control
import matplotlib.pyplot as plt

def callbackFunc(xk, state):
    callbackFunc.epoch += 1
    print("{}: Jfrit = {:4.2f}, rho = {}".format(
        callbackFunc.epoch, 
        state["fun"], 
        np.round(xk, 2)))
    callbackFunc.ax1.scatter(callbackFunc.epoch, state["fun"])
    if callbackFunc.epoch % 2 == 0:
        plt.pause(.01)

class gain_estimator:
    def pid(self, rho):
        s = control.tf([1, -1], [1, 1], self.dt) * 2 / self.dt # z transform
        return rho[0] + rho[1] / s + rho[2] * s

    def pdf(self, rho):
        s = control.tf([1, -1], [1, 1], self.dt) * 2 / self.dt # z transform
        return rho[0] + rho[1] * s / (rho[2] * s + 1)

    def select_model(self, model, dt):
        if model == 'pid':
            self.ctrl_func = self.pid
        elif model == 'pdf':
            self.ctrl_func = self.pdf
        self.dt = dt

    def eval_func(self, rho, u0, y0, Td, W):
        invC = self.ctrl_func(rho) ** -1
        u_filt, _, _ = control.lsim(W, u0.flatten())
        y_filt, _, _ = control.lsim(W, y0.flatten())
        e_pseudo, _, _ = control.lsim(invC, u_filt.flatten())
        r_pseudo = e_pseudo + y_filt
        y_pseudo, _, _ = control.lsim(Td, r_pseudo.flatten())
        return np.linalg.norm(y_filt - y_pseudo)**2

    def estimate(self, rho0, input, output, Td, W):
        from scipy.optimize import minimize
        from scipy.optimize import LinearConstraint, BFGS

        callbackFunc.epoch = 0
        callbackFunc.fig1 = plt.figure()
        callbackFunc.ax1 = callbackFunc.fig1.add_subplot(1, 1, 1)
        cons = LinearConstraint(
            np.eye(rho0.size), 
            np.zeros_like(rho0), 
            np.full_like(rho0, np.inf))

        ret = minimize( fun=self.eval_func,
                        x0 = rho0,
                        args = (input, output, Td, W),
                        method="trust-constr", 
                        jac="2-point",
                        hess=BFGS(),
                        constraints=cons,
                        options={"maxiter":np.inf, "disp":True},
                        callback=callbackFunc
                    )
        plt.show()
        return ret["x"]
        

if __name__=="__main__":
    from model_creator import *
    mc = model_creator()
    mc.create_model_1st(1.02, 0.74)
    #mc.create_model_2nd(1.02, 20, 0.8)
    u = np.ones(1000)
    dt = 0.001
    y, t = mc.generate_dataset(u, dt, False, False)

    ge = gain_estimator()
    ge.select_model('pid', dt)
    rho0 = np.array([10.0, 0.0, 0.0])
    tau = 0.02
    Td = control.c2d(control.tf([1], [tau, 1]), dt)
    fc = 5.0
    W = control.c2d(control.tf([1], [1 / (2 * np.pi * fc), 1]), dt)
    rho = ge.estimate(rho0, u, y, Td, W)
    print('rho = ',rho)
    C = ge.ctrl_func(rho)
    plt.figure()
    control.bode(C)
    plt.show()