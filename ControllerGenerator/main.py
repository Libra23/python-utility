#!/usr/bin/env python3

from gain_estimator import *
from model_creator import *

mc = model_creator()
mc.create_model_1st(1.02, 0.74)
u = np.ones(1000)
dt = 0.001
y, t = mc.generate_dataset(u, dt, False, False)

