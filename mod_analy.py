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

# %%
import os
import json
import warnings
import sys
import matplotlib
from matplotlib import colors
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import optimize
from sklearn import metrics
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import mod_steady_shape
from tqdm import tqdm

matplotlib.style.use("tj_origin.mplstyle")

# %%%
class Fitting:
    """Class for fitting several model constants to experimental results,
    and output some figures.

    Attributes
    ----------
    param: dict
        dictionary of setting of fitting calculation
    fldlist: list of string
        list of the folder name of experimental results
    exp_cond: dict
        dictionary of experimental condition for each experiment
    shape_dat: dict of pandas.DataFrame
        dictionary of regressioin shape
    """
    def __init__(self, cond):
        """ Constructor
        
        Parameters
        ----------
        cond : dict
            dictionary of setting for fitting calculation
        """
        self.param = cond
        if "fldlist" in cond:
            self.fldlist = cond["fldlist"]
        else:
            self.fldlist = self._get_fldlist_()
        self.exp_cond, self.shape_dat = self._read_dat_(self.fldlist)

    def _get_fldlist_(self):
        """function to get the list of experimental folder name
        
        Returns
        -------
        fldlist: list of string
            dictionary of experimental folder name
        """
        fldlist = os.listdir(path="exp_dat")
        if "sample" in fldlist:
            if len(fldlist) == 1:
                print("Please make folder whose name shoud not be \"sample\" and contains data files.")
                sys.exit()
            else:
                fldlist.remove("sample")
        else:
            pass
        return fldlist

    def _read_dat_(self, fldlist):
        """ function to read each experimental condition and data of regression shape
        
        Parameters
        ----------
        fldlist : list of string
            list of experimental folder name
        
        Returns
        -------
        exp_cond: dict of dict
            dictionary of each experimental condition
        shape_dat: dict of pandas.DataFrame
            dictionary of fuel regression shape data
        """
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
        """ Calculate the mean R or R2 for all experimental data.
        
        Parameters
        ----------
        Cr : float
            proportional constant of fuel regression model
        z : float
            mass flux exponent constant of fuel regression model
        m : float
            axial direction exponent constant of fuel regression model
        mode : str, optional
            mode selection for output value whether "R" or "R2", by default "R2"
        
        Returns
        -------
        coef.mean()
            mean value of coefficient
        """
        coef = np.array([])
        for testname, exp_cond in self.exp_cond.items():
            tmp = self.get_R_R2(testname, Cr, z, m, mode=mode)
            coef = np.append(coef, tmp)
        return coef.mean()

    def get_R_R2(self, testname, Cr, z, m, mode="R2"):
        """ Calculate the R or R2 for assgined experimental data.
        
        Parameters
        ----------
        testname : str
            experimental name, which is the same of folder name.
        Cr : float
            proportional constant of fuel regression model
        z : float
            mass flux exponent constant of fuel regression model
        m : float
            axial direction exponent constant of fuel regression model
        mode : str, optional
            mode selection for output value whether "R" or "R2", by default "R2"
        
        Returns
        -------
        coef: float
            the value of coefficient. "R" or "R2"
        """
        x = np.array(self.shape_dat[testname].x)
        r = np.array(self.shape_dat[testname].r)
        exp_cond = self.exp_cond[testname]
        Pc = exp_cond["Pc"]
        Vox = exp_cond["Vox"]
        d = exp_cond["d"]
        inst = mod_steady_shape.Main(Pc, Vox, **self.param, x_max=x.max(), d=d, Cr=Cr, z=z, m=m)
        x_model, r_model, rdot_model = inst.exe()
        func_interp = interpolate.CubicSpline(x_model, r_model, bc_type="natural", extrapolate=None)
        r_interp = np.array([func_interp(val) for val in x])
        if mode == "R2":    # calaculate coefficient of derming factor, R2.
            coefficient = metrics.r2_score(r, r_interp)
        elif mode == "R":   # calculate correlate coefficient, R.
            coefficient = np.corrcoef(r, r_interp)[0][1]
        else:
            warnings.warn("Please assign proper keyword argument. Not \"{}\" but \"R\" or \"R2\"".format(mode))
            sys.exit()
        return coefficient

    def optimize_modelconst(self, mode="R2", **kwargs):
        """ Optimization for model constants, Cr, z, and m.
        
        Parameters
        ----------
        mode : str, optional
            mode selection for output value whether "R" or "R2", by default "R2"
        
        Returns
        -------
        res: the output object of scipy.optimize.minimize
            "res" contains the list of optimized constans and minimized coefficient so on.
            Please look the page of scipy documentation for optimize.minimize
        """
        print("Now optimizing model costants. Please wait a minute.")
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
            if method == "global":      # optimization for seeking global mimimum
                bounds = kwargs["bounds"]
                res = optimize.differential_evolution(func_opt, bounds=bounds, args=(mode,))
            else:                       # optimizatioin for seeking local minimum with assigning optimization method
                res = optimize.minimize(func_opt, x0=init_const, args=(mode,), method=method)
        else:                           # optimizaiton for seeking local minimum
            res = optimize.minimize(func_opt, x0=init_const, args=(mode,))
        print("Finish optimization of model constants!")
        return res

    def gen_excomp_figlist(self, Cr, z, m, mode="R2"):
        """ Generate the figures which compare the experimental result and calculation result obtained by assigned constants.
        
        Parameters
        ----------
        testname : str
            experimental name, which is the same of folder name.
        Cr : float
            proportional constant of fuel regression model
        z : float
            mass flux exponent constant of fuel regression model
        m : float
            axial direction exponent constant of fuel regression model
        mode : str, optional
            mode selection for output value whether "R" or "R2", by default "R2"
        
        Returns
        -------
        dic: dict of matplotlib.pyplot.figure
            dictionary of matplotlib figures
        """
        xmax = 0.0
        rmax = 0.0
        for testname in self.exp_cond:
            xtmp = np.array(self.shape_dat[testname].x).max()
            if xmax < xtmp:
                xmax = xtmp
            rtmp = np.array(self.shape_dat[testname].r).max()
            if rmax < rtmp:
                rmax = rtmp
        dic={}
        for testname in self.exp_cond:
            dic_tmp={}
            dic_tmp["coef"] = self.get_R_R2(testname ,Cr, z, m, mode=mode)
            dic_tmp["fig"] = self.plot_expcomp(testname, Cr, z, m, dic_tmp["coef"], mode=mode, xmax=xmax, rmax=rmax)
            dic[testname] = dic_tmp
        return dic

    def plot_expcomp(self, testname, Cr, z, m, r_r2, mode="R2", xmax=None, rmax=None):
        """ plot a figures which comaper the regressionshape of experimental result and calcultion.
        
        Parameters
        ----------
        testname : str
            experimental name, which is the same of folder name.
        Cr : float
            proportional constant of fuel regression model
        z : float
            mass flux exponent constant of fuel regression model
        m : float
            axial direction exponent constant of fuel regression model
        r_r2 : float
            coefficient for write on a figure
        mode : str, optional
            mode selection for output value whether "R" or "R2", by default "R2"
        xmax: float, optional
            x limit of figure x axis, by default None. If None, code uses maximum x of experiment
        rmax: float, optional
            r limit of figure y axis, by default None. If None, code uses maximum r of experiment
        
        Returns
        -------
        fig: matplotlib.pyplot.figure
            figure object of matplotlib
        """
        x = np.array(self.shape_dat[testname].x)
        r = np.array(self.shape_dat[testname].r)
        exp_cond = self.exp_cond[testname]
        Pc = exp_cond["Pc"]
        Vox = exp_cond["Vox"]
        d = exp_cond["d"]        
        inst = mod_steady_shape.Main(Pc, Vox, **self.param, x_max=x.max(), d=d, Cr=Cr, z=z, m=m)
        x_model, r_model, rdot_model = inst.exe()
        fig = plt.figure(figsize=(12,9))
        ax = fig.add_subplot(111)
        ax.plot(x*1e+3, r*1e+3, color="r", label="Experiment")
        ax.plot(x_model*1e+3, r_model*1e+3, color="b", label="Calculation")
        if rmax is None:
            rmax = r.max()
        else:
            pass
        if xmax is None:
            xmax = x.max()
        else:
            pass
        ax.set_ylim(0, rmax*1e+3)
        ax.set_xlim(0, xmax*1e+3)
        if mode == "R2":
            text = "R2"
        else:
            text= "R"
        ax.text(0.05*xmax*1e+3, 0.9*rmax*1e+3, "{}={}".format(text, round(r_r2,3)), fontsize=30)
        ax.text(0.05*xmax*1e+3, 0.75*rmax*1e+3, \
            "$P_c$= {} MPa, $d$= {} mm,".format(round(Pc*1e-6,3), round(d*1e+3,2))\
            +"\n$V_{ox}$"+"= {} m/s".format(round(Vox,1)), fontsize=25)
        ax.set_title(testname, fontsize=40)
        ax.legend(loc="lower right")
        ax.set_xlabel("Axial distance $x$ [mm]")
        ax.set_ylabel("Radial regression distance $r$ [mm]")
        return fig


    
    def gen_R_R2_figlist(self, bounds, mode="R2", resolution=10, thirdparam="Cr", num_fig=9, **kwargs):
        bound_Cr = bounds[0]
        bound_z = bounds[1]
        bound_m = bounds[2]
        if thirdparam == "Cr":
            interb_z = (bound_z[1] - bound_z[0])/resolution
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
                    if thirdparam == "Cr":
                        coef = self.get_R_R2_mean(Cr=third_array[i], z=x_array[j], m=y_array[k], mode=mode)
                    elif thirdparam == "z":
                        coef = self.get_R_R2_mean(Cr=y_array[k], z=third_array[i], m=x_array[j], mode=mode)
                    else:
                        coef = self.get_R_R2_mean(Cr=y_array[j], z=x_array[j], m=third_array[i], mode=mode)
                    z[i, j, k] = coef

        # self._plot_(x_array, y_array, z, thirdparam="Cr", num_fig=num_fig)
        return x_array, y_array, z

    def plot_R_R2_contour(self, x, y, z, thirdparam="Cr", num_fig=9):
        fig = plt.figure(figsize=(24,24))
        ax = fig.add_subplot(1,1,1, projection="3d")
        # ax = [0 for i in range(num_fig)]
        # for i in range(num_fig):
        #     ax[i] = fig.add_subplot(num_fig, num_fig, i)
        X, Y = np.meshgrid(x, y)
        norm_bound = plt.Normalize(0.0, 1.0)
        cmap = cm.plasma
        cmap.set_under((0,0,0,0), alpha=0.0)
        surf = ax.plot_surface(X, Y, z[0], rstride=1, cstride=1, cmap=cmap)
        cbar = fig.colorbar(surf, shrink=0.75)
        cbar.set_label("R2")
        ax.contour(X, Y, z[0], levels=10, norm=norm_bound, cmap=cmap)
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_zlim(0, 1.0)
        fig.savefig("test.png", dpi=400)

       
# %%
def plot(x, y, z, thirdparam="Cr"):
    fig = plt.figure(figsize=(30,24))
    ax = fig.add_subplot(1,1,1, projection="3d")
    # ax = [0 for i in range(num_fig)]
    # for i in range(num_fig):
    #     ax[i] = fig.add_subplot(num_fig, num_fig, i)
    X, Y = np.meshgrid(x, y)
    surface_alpha_max = 0.5
    surface_alpha_min = 0.0
    norm_bound = plt.Normalize(vmin=0.0, vmax=1.0)
    cmap = cm.coolwarm
    cmap_surface = cmap(np.arange(cmap.N))
    cmap_surface[:,-1] = np.linspace(surface_alpha_min, surface_alpha_max, cmap.N)
    cmap_surface = colors.ListedColormap(cmap_surface)
    cmap_surface.set_under((0,0,0,0), alpha=0.0)
    cmap.set_under((0,0,0,0), alpha=0.0)
    surf = ax.plot_surface(X, Y, z[0], rstride=1, cstride=1, norm=norm_bound, cmap=cmap_surface)
    cbar = fig.colorbar(surf, shrink=0.75)
    cbar.set_label("R2", fontsize=50)
    cbar.ax.tick_params(labelsize=40)
    cntr_z = ax.contour(X, Y, z[0], \
        levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], \
        zdir="z", offset=0.0, norm=norm_bound, cmap=cmap)
    cntr_z.clabel(inline=1, colors="k", fontsize=5, use_clabeltext=True)
    index_xmax, index_ymax = np.unravel_index(np.argmax(z[0]), z[0].shape)
    maximum = (X[index_xmax, index_ymax], Y[index_xmax, index_ymax], z[0].max())
    xmax = maximum[0]
    ymax = maximum[1]
    zmax = maximum[2]
    ax.plot([xmax,x.max()], [ymax, ymax], [0,0], c="k")
    ax.plot([xmax,xmax], [y.min(), ymax], [0,0], c="k")
    ax.plot([xmax,xmax], [ymax, ymax], [0,zmax], c="k")
    ax.plot([xmax], [ymax], [zmax], marker="o", ms=30, c="k")
    ax.text(xmax, y.min()-0.12, 0, \
        "{}={}".format("z", round(xmax,4)), "x", zorder=10, fontsize=50)
    ax.text(x.max()+0.1, ymax-0.05, 0, \
        "{}={}".format("m", round(ymax,4)), "y", zorder=10, fontsize=50)
    ax.text(xmax, ymax, 1.1*zmax, \
        "R2$_{max}$"+"={}".format(round(zmax,4)), zorder=10, fontsize=60)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(0, 1.0)
    ax.set_zlabel("$R2$", fontsize=50)
    ax.tick_params(axis="both", labelsize=40)
    fig.suptitle("$C_r$={}".format("***"), fontsize=80)
    # fig.savefig("test.png", dpi=400)
    return fig




# %%





if __name__ == "__main__":
    # %%
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
            #  "x_max": 12.6e-3,  # [m] maximum calculation region
             "r_0": 0.0,        # r=0.0 when x = 0, boudanry condition
             "rdot_0": 0.0,     # rdot=0 when x = 0, boundary condition
             "Vf_mode": False , # mode selection using axial or radial integretioin for mf calculation
             "use_Vf": False    # mode selection for using experimental Vf insted of Vf empirical formula or not.
            }
    
    BOUND = {"Cr": (1.0e-6, 30.0e-6),
             "z": (0.0, 1.0),
             "m": (-0.5, 0.0)
             }

    CONTOUR_PLOT = {"Cr_bnd": (14.0e-6, 16.0e-6),      # plot range of Cr
                    "z_bnd": (0.2, 0.6),               # plot range of z
                    "m_bnd": (-0.4, -0.1),             # plot range of m
                    "resol": 100,                      # the number of calculating point for x and y axis direction
                    "thirdparam": "Cr",                # select the thrid parameter. selected parameter is varied with the number of "num_fig"
                    "num_fig": 1                       # the number of figures. This value is the same of the number of variety of "thirdparam".
                    }
    
    # %%
    # Temporal Code to Debug the function of plot
    # inst = Fitting(**PARAM)
    # x_array, y_array, z_array = inst.plot_R_R2(bounds=[(14e-6, 16e-6), (0.2, 0.6), (-0.4, -0.1)],\
    #      mode="R2", resolution=100, thirdparam="Cr", num_fig=1)
    # plot(x_array, y_array, z_array, thirdparam="Cr", num_fig=1)

    # %%
    inst = Fitting(PARAM)
    ## optimization for model constants
    RES_TMP = inst.optimize_modelconst(mode="R2", method="global", bounds=[(1.0e-6, 30.0e-6), (0.0, 1.0), (-0.5, 0.0)])
    RESULT = {"Cr": RES_TMP.x[0],
              "z": RES_TMP.x[1],
              "m": RES_TMP.x[2],
              "R2mean": -RES_TMP.fun,
              "success": RES_TMP.success
              }
    ## calculate R2 for each experiment    
    R2 = {}
    for testname in inst.exp_cond:
        R2[testname] = inst.get_R_R2(testname, RESULT["Cr"], RESULT["z"], RESULT["m"], mode="R2")
    
    OUTPUT = {"cond": PARAM,
              "bounds": BOUND,
              "contour_plot": CONTOUR_PLOT,
              "result": RESULT,
              "R2": R2
              }
    ## output calculation condition and result
    FLDNAME = datetime.now().strftime("%Y_%m%d_%H%M%S")
    os.mkdir(FLDNAME)
    with open(os.path.join(FLDNAME, "result.json"), "w") as f:
        json.dump(OUTPUT, f, ensure_ascii=False, indent=4)
    ## output figures which compare experimental and calculated result
    FLDNAME_EXPCOMP = "fig_expcomp"
    os.mkdir(os.path.join(FLDNAME, FLDNAME_EXPCOMP))
    FIG_COMP = inst.gen_excomp_figlist(RESULT["Cr"], RESULT["z"], RESULT["m"], mode="R2")
    for testname, dic in FIG_COMP.items():
        dic["fig"].savefig(os.path.join(FLDNAME, FLDNAME_EXPCOMP, "{}.png".format(testname)), dpi=300)
    
    # %%
    # print("Cr={}, z={}, m={}, R2={}".format(res.x[0], res.x[1], res.x[2], -res.fun))
    print("Compleated!")

# %%
