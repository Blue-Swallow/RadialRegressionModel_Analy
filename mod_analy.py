# coding -*- utf-8 -*-
""" 
Autor: Ayumu Tsuji
Date: 20200821

Description:
This file is a module for analyzing fuel regression shape of single port.
And calculate several proportional constant and exponet constants of
radial fuel regrerssion model, which is resemble with conventional
cylindrical hybrid rocket.
"""

import os
import json
import warnings
import sys
import matplotlib
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import optimize
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import mod_steady_shape
from tqdm import tqdm

matplotlib.style.use("tj_origin.mplstyle")

class Fitting:
    def __init__(self, **kwargs):
        self.param = kwargs
        if "fldlist" in kwargs:
            self.fldlist = kwargs["fldlist"]
        else:
            self.fldlist = self._get_fldlist_()
        self.exp_cond, self.shape_dat = self._read_dat_(self.fldlist)

    def _get_fldlist_(self):
        fldlist = os.listdir(path="exp_dat")
        if "sample" in fldlist:
            if len(fldlist) == 1:
                print("Please make folder which name shoud not be \"sample\" and contains data files.")
                sys.exit()
            else:
                fldlist.remove("sample")
        else:
            pass
        return fldlist

    def _read_dat_(self, fldlist):
        exp_cond = {}
        shape_dat = {}
        for fldname in fldlist:
            with open(os.path.join("exp_dat", fldname, "exp_cond.json")) as f:
                tmp = json.load(f)
                tmp["Pc"] = tmp["Pc"]*1e+6      # convert unit MPa to Pa
                tmp["d"] = tmp["d"]*1e-3        # convert unit mm to m
                exp_cond[fldname] = tmp
            tmp = pd.read_csv(os.path.join("exp_dat", fldname, "shape_dat.csv"), header=0, skiprows=[1,])
            tmp.x = tmp.x*1e-3  # convert unit mm to m
            tmp.r = tmp.r*1e-3  # convert unit mm to m
            shape_dat[fldname] = tmp
        return exp_cond, shape_dat

    def get_R_R2_mean(self, Cr, z, m, mode="R2"):
        coef = np.array([])
        for testname, exp_cond in self.exp_cond.items():
            x = np.array(self.shape_dat[testname].x)
            r = np.array(self.shape_dat[testname].r)
            Pc = exp_cond["Pc"]
            Vox = exp_cond["Vox"]
            d = exp_cond["d"]
            tmp = self.get_R_R2(x, r, Pc, Vox, d, Cr, z, m, mode=mode)
            coef = np.append(coef, tmp)
        return coef.mean()

    def get_R_R2(self, x, r, Pc, Vox, d, Cr, z, m, mode="R2"):
        inst = mod_steady_shape.Main(Pc, Vox, **self.param, d=d, Cr=Cr, z=z, m=m)
        x_model, r_model, rdot_model = inst.exe()
        func_interp = interpolate.CubicSpline(x_model, r_model, bc_type="natural", extrapolate=None)
        r_interp = np.array([func_interp(val) for val in x])
        if mode is "R2":    # calaculate coefficient of derming factor, R2.
            coefficient = metrics.r2_score(r, r_interp)
        elif mode is "R":   # calculate correlate coefficient, R.
            coefficient = np.corrcoef(r, r_interp)[0][1]
        else:
            warnings.warn("Please assign proper keyword argument. Not \"{}\" but \"R\" or \"R2\"".format(mode))
            sys.exit()
        return coefficient

    def optimize_modelconst(self, mode="R2", **kwargs):
        if "Cr_init" in kwargs:
            Cr_init = kwargs["Cr_init"]
        else:
            Cr_init = self.param["Cr_init"]
        if "z_init" in kwargs:
            z_init = kwargs["z_init"]
        else:
            z_init = self.param["z_init"]
        if "m_init" in kwargs:
            m_init = kwargs["m_init"]
        else:
            m_init = self.param["m_init"]
        init_const = np.array([Cr_init, z_init, m_init])
        func_opt = lambda const, mode: -1*self.get_R_R2_mean(const[0], const[1], const[2], mode=mode)
        if "method" in kwargs:
            method = kwargs["method"]
            if method is "global":      # optimization for seeking global mimimum
                bounds = kwargs["bounds"]
                res = optimize.differential_evolution(func_opt, bounds=bounds, args=(mode,))
            else:                       # optimizatioin for seeking local minimum with assigning optimization method
                res = optimize.minimize(func_opt, x0=init_const, args=(mode,), method=method)
        else:                           # optimizaiton for seeking local minimum
            res = optimize.minimize(func_opt, x0=init_const, args=(mode,))
        return res
    
    def plot_R_R2(self, bounds, mode="R2", resolution=10, thirdparam="Cr", num_fig=9, **kwargs):
        bound_Cr = bounds[0]
        bound_z = bounds[1]
        bound_m = bounds[2]
        if thirdparam == "Cr":
            interb_z = (bound_z[1] - bound_z[0])/resolution
            x_array = np.arange(bound_z[0], bound_z[1]+interb_z/2, interb_z)
            interb_m = (bound_m[1] - bound_m[0])/resolution
            y_array = np.arange(bound_m[0], bound_m[1]+interb_m/2, interb_m)
            if num_fig == 1:
                third_array = np.array([(bound_Cr[0]+bound_Cr[1])/2])
            else:
                interb_Cr = (bound_Cr[1] - bound_Cr[0])/(num_fig-1)
                third_array = np.arange(bound_Cr[0], bound_Cr[1]+interb_Cr/2, interb_Cr)
        elif thirdparam == "z":
            if num_fig == 1:
               third_array = np.array([(bound_z[0]+bound_z[1])/2])
            else:
                interb_z = (bound_z[1] - bound_z[0])/(num_fig-1)
                third_array = np.arange(bound_z[0], bound_z[1]+interb_z/2, interb_z)
            interb_m = (bound_m[1] - bound_m[0])/resolution
            x_array = np.arange(bound_m[0], bound_m[1]+interb_m/2, interb_m)
            interb_Cr = (bound_Cr[1] - bound_Cr[0])/resolution
            y_array = np.arange(bound_Cr[0], bound_Cr[1]+interb_Cr/2, interb_Cr)
        elif thirdparam == "m":
            interb_z = (bound_z[1] - bound_z[0])/resolution
            x_array = np.arange(bound_z[0], bound_z[1]+interb_z/2, interb_z)
            if num_fig == 1:
               third_array = np.array([(bound_m[0]+bound_m[1])/2])
            else:
                interb_m = (bound_m[1] - bound_m[0])/(num_fig-1)
                third_array = np.arange(bound_m[0], bound_m[1]+interb_m/2, interb_m)
            interb_Cr = (bound_Cr[1] - bound_Cr[0])/resolution
            y_array = np.arange(bound_Cr[0], bound_Cr[1]+interb_Cr/2, interb_Cr)
        else:
            print("There is no such a parameter thirdparam=\"{}\"".format(thirdparam))
            sys.exit()
        z = np.empty((num_fig, resolution+1, resolution+1))
        for i in tqdm(range(len(third_array)), desc="Total Progress", leave=True):
            for j in tqdm(range(len(x_array)), desc="Cr={}".format(third_array[i]), leave=False):
                for k in range(len(y_array)):
                    if thirdparam is "Cr":
                        coef = self.get_R_R2_mean(Cr=third_array[i], z=x_array[j], m=y_array[k], mode=mode)
                    elif thirdparam is "z":
                        coef = self.get_R_R2_mean(Cr=y_array[k], z=third_array[i], m=x_array[j], mode=mode)
                    else:
                        coef = self.get_R_R2_mean(Cr=y_array[j], z=x_array[j], m=third_array[i], mode=mode)
                    z[i, j, k] = coef

        self._plot_(x_array, y_array, z, thirdparam="Cr", num_fig=num_fig)
        return z

    def _plot_(self, x, y, z, thirdparam="Cr", num_fig=9):
        fig = plt.figure(figsize=(24,24))
        ax = fig.add_subplot(1,1,1, projection="3d")
        # ax = [0 for i in range(num_fig)]
        # for i in range(num_fig):
        #     ax[i] = fig.add_subplot(num_fig, num_fig, i)
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, z[0], alpha=0.3, vmin=0.0, vmax=1.0, cmap=cm.cividis)
        ax.contour(X, Y, z[0], levels=10, vmin=0.0, vmax=1.0, cmap=cm.cividis)
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_zlim(0, 1.0)
        fig.savefig("test.png", dpi=400)









if __name__ == "__main__":
    PARAM = {"rho_f": 1190,     # [kg/m^3] solid fuel density
             "M_ox": 32.0e-3,   # [kg/mol]
             "T": 300,          # [K] oxidizer tempreature
             "Cr_init": 15e-6,# Initial guess value for coefficent fitting
             "z_init": 0.4,     # Initial guess value for coefficent fitting
             "m_init": -0.26,   # Initial guess value for coefficent fitting
             "C1": 1.39e-7,  # experimental constant of experimental regression rate formula
             "C2": 1.61e-9,  # experimental constant of experimental regression rate formula
             "n": 1.0,       # experimental exponent constant of pressure
             "dx": 0.1e-3,      # [m] space resolution
             "x_max": 12.6e-3,  # [m] maximum calculation region
             "r_0": 0.0,        # r=0.0 when x = 0, boudanry condition
             "rdot_0": 0.0,     # rdot=0 when x = 0, boundary condition
             "Vf_mode": False   # mode selection using axial or radial integretioin for mf calculation
            }
    inst = Fitting(**PARAM)
    Cr = 15.0e-6
    z = 0.4
    m = -0.26
    coef = inst.get_R_R2_mean(Cr, z, m, mode="R2")
    print("Coefficient = {}".format(coef))
    z_array = inst.plot_R_R2(bounds=[(14e-6, 16e-6), (0.2, 0.6), (-0.4, -0.1)], mode="R2", resolution=10, thirdparam="Cr", num_fig=1)
    # print(z_array)
    # res = inst.optimize_modelconst(mode="R2", method="TNC")
    # res = inst.optimize_modelconst(mode="R2", method="global", bounds=[(1.0e-6, 30.0e-6), (0.0, 1.0), (-0.5, 0.0)])
    # res = inst.optimize_modelconst(mode="R", method="global", bounds=[(1.0e-6, 100.0e-6), (0.0, 1.0), (-0.5, 0.0)])
    # print("Cr={}, z={}, m={}, R2={}".format(res.x[0], res.x[1], res.x[2], -res.fun))
