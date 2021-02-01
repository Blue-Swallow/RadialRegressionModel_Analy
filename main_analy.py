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
import copy
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
    def __init__(self, cond, fldpath):
        """ Constructor
        
        Parameters
        ----------
        cond : dict
            dictionary of setting for fitting calculation
        fldpath : string
            folder path which contains "cond.py" and each experimental data folder.
        """
        self.param = cond
        self.fldpath = fldpath
        if "fldlist" in cond:
            self.fldlist = cond["fldlist"]
        else:
            self.fldlist = self._get_fldlist_(self.fldpath)
        self.exp_cond, self.shape_dat = self._read_dat_(self.fldlist, fldpath)

    def _get_fldlist_(self, fldpath):
        """function to get the list of experimental folder name
        
        Returns
        -------
        fldlist: list of string
            dictionary of experimental folder name
        """
        # fldlist = os.listdir(path="exp_dat")
        fldlist = os.listdir(path=fldpath)
        fldlist = [f for f in fldlist if os.path.isdir(os.path.join(fldpath, f))] # eliminate the list of files
        fldlist.remove("__pycache__") # eliminate chach folder path
        fldlist = [txt for txt in fldlist if "sample" not in txt] # eliminate sample folder
        fldlist = [txt for txt in fldlist if "fig_" not in txt] # eliminate result folder
        if len(fldlist) == 0:
            print("Please make folder whose name shoud not be \"sample*\" and contains data files.")
            sys.exit()
        else:
            pass
        return fldlist

    def _read_dat_(self, fldlist, fldpath):
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
            # with open(os.path.join("exp_dat", fldname, "exp_cond.json")) as f:
            with open(os.path.join(fldpath, fldname, "exp_cond.json")) as f:
                tmp = json.load(f)
                tmp["Pc"] = tmp["Pc"]*1e+6      # convert unit MPa to Pa
                tmp["d"] = tmp["d"]*1e-3        # convert unit mm to m
                if tmp["Vf"] is not None:
                    tmp["Vf"] = tmp["Vf"]*1e-3  # convert unit mm to m
                exp_cond[fldname] = tmp
            # tmp = pd.read_csv(os.path.join("exp_dat", fldname, "shape_dat.csv"), header=0, skiprows=[1,])
            tmp = pd.read_csv(os.path.join(fldpath, fldname, "shape_dat.csv"), header=0, skiprows=[1,])
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
        if exp_cond["Vf"] is None:
            tmp_param = copy.deepcopy(self.param)
            tmp_param["use_Vf"] = False
            obj = mod_steady_shape.Main(Pc, Vox, **tmp_param, x_max=x.max(), d=d, Cr=Cr, z=z, m=m)
        else:
            obj = mod_steady_shape.Main(Pc, Vox, **self.param, x_max=x.max(), d=d, Cr=Cr, z=z, m=m, Vf=exp_cond["Vf"])
        x_model, r_model, rdot_model = obj.exe()
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
                rmax = rtmp *1.2
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
        if exp_cond["Vf"] is None:
            Vf_ex = "null"
            tmp_param = copy.deepcopy(self.param)
            tmp_param["use_Vf"] = False
            obj = mod_steady_shape.Main(Pc, Vox, **tmp_param, x_max=x.max(), d=d, Cr=Cr, z=z, m=m)
        else:
            Vf_ex = exp_cond["Vf"]
            obj = mod_steady_shape.Main(Pc, Vox, **self.param, x_max=x.max(), d=d, Cr=Cr, z=z, m=m, Vf=exp_cond["Vf"])
        Vf_th = obj.func_Vf(Vox, Pc)
        x_model, r_model, rdot_model = obj.exe()
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
        Vf_th_text = round(Vf_th*1e+3, 2)
        if type(Vf_ex) is float:
            Vf_ex_text = round(Vf_ex*1e+3, 2)
        ax.text(0.05*xmax*1e+3, 0.9*rmax*1e+3, "{}={}".format(text, round(r_r2,3)), fontsize=30)
        ax.text(0.05*xmax*1e+3, 0.72*rmax*1e+3, \
            "$P_c$= {} MPa, $d$= {} mm,".format(round(Pc*1e-6,3), round(d*1e+3,2))\
            +"$V_{ox}$"+"= {} m/s".format(round(Vox,1))\
            +"\n$V_{f,ex}$"+"= {} mm/s".format(Vf_ex_text) + ", $V_{f,th}$"+"= {} mm/s".format(Vf_th_text), fontsize=25)
        ax.set_title(testname, fontsize=40)
        ax.legend(loc="lower right")
        ax.set_xlabel("Axial distance $x$ [mm]")
        ax.set_ylabel("Radial regression distance $r$ [mm]")
        return fig

    def gen_R_R2_figlist(self, bounds, mode="R2", resolution=50, thirdparam="Cr", num_fig=9, fldname=None, **kwargs):
        z_array, x_array, y_array, third_array = self.cal_R_R2_contour(bounds=bounds, mode=mode,\
            resolution=resolution, thirdparam=thirdparam, num_fig=num_fig, fldname=fldname, **kwargs)
        fig_list = []
        for z, third in zip(z_array, third_array):
            fig, xmax, ymax, zmax = self.plot_R_R2_contour(x_array, y_array, z, third, thirdparam=thirdparam, mode=mode)
            dic = {"third": third,
                   "fig": fig,
                   "xmax": xmax,
                   "ymax": ymax,
                   "zmax": zmax
                  }
            fig_list.append(dic)
        return fig_list
            

    def cal_R_R2_contour(self, bounds, mode="R2", resolution=50, thirdparam="Cr", num_fig=9, fldname=None, **kwargs):
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
            for j in tqdm(range(len(x_array)), desc="{0}={1:6.3e}".format(thirdparam, third_array[i]), leave=False):
                for k in tqdm(range(len(y_array)), leave=False):
                    if thirdparam == "Cr":
                        coef = self.get_R_R2_mean(Cr=third_array[i], z=x_array[j], m=y_array[k], mode=mode)
                    elif thirdparam == "z":
                        coef = self.get_R_R2_mean(Cr=y_array[k], z=third_array[i], m=x_array[j], mode=mode)
                    else:
                        coef = self.get_R_R2_mean(Cr=y_array[j], z=x_array[j], m=third_array[i], mode=mode)
                    z[i, j, k] = coef
            if fldname is None:
                pass
            else:
                df = pd.DataFrame(z[i], columns=x_array, index=y_array)
                df.to_csv(os.path.join(fldname, "{0}={1:5.3e}.csv".format(thirdparam, third_array[i])))
        return z, x_array, y_array, third_array

    def plot_R_R2_contour(self, x, y, z, val_thirdparam, thirdparam="Cr", mode="R2"):
        if thirdparam == "Cr":
            xaxis = "z"
            yaxis = "m"
        elif thirdparam == "z":
            xaxis = "m"
            yaxis = "Cr"
        elif thirdparam == "m":
            xaxis = "z"
            yaxis = "Cr"
        else:
            print("There is no such a parameter thirdparam=\"{}\"".format(thirdparam))
            sys.exit()
        fig = plt.figure(figsize=(30,24))
        ax = fig.add_subplot(1,1,1, projection="3d")
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
        surf = ax.plot_surface(X, Y, z, rstride=1, cstride=1, norm=norm_bound, cmap=cmap_surface)
        cbar = fig.colorbar(surf, shrink=0.75)
        cbar.set_label("R2", fontsize=50)
        cbar.ax.tick_params(labelsize=40)
        cntr_z = ax.contour(X, Y, z, \
            levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], \
            zdir="z", offset=0.0, norm=norm_bound, cmap=cmap)
        cntr_z.clabel(inline=1, colors="k", fontsize=5, use_clabeltext=True)
        index_xmax, index_ymax = np.unravel_index(np.argmax(z), z.shape)
        xmax = X[index_xmax, index_ymax]
        ymax = Y[index_xmax, index_ymax]
        zmax = z.max()
        ax.plot([xmax,x.max()], [ymax, ymax], [0,0], c="k")
        ax.plot([xmax,xmax], [y.min(), ymax], [0,0], c="k")
        ax.plot([xmax,xmax], [ymax, ymax], [0,zmax], c="k")
        ax.plot([xmax], [ymax], [zmax], marker="o", ms=30, c="k")
        ax.text(xmax, y.min()-(y.max()-y.min())*0.5, 0, \
            "{0}={1:6.3e}".format(xaxis, xmax), "x", zorder=10, fontsize=50)
        ax.text(x.max()+(x.max()-x.min())*0.2, ymax, 0, \
            "{0}={1:6.3e}".format(yaxis, ymax), "y", zorder=10, fontsize=50)
        ax.text(xmax, ymax, 1.1*zmax, \
            mode+"$_{max}$"+"={}".format(round(zmax,4)), zorder=10, fontsize=60)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_zlim(0, 1.0)
        ax.set_zlabel(mode, fontsize=50)
        ax.tick_params(axis="both", labelsize=40)
        fig.suptitle("{0}={1:4.2e}".format(thirdparam, val_thirdparam), fontsize=80)
        return fig, xmax, ymax, zmax

def read_cond():
    """ Function to assign a folder and read condition file: "cond.py".
    
    Returns
    -------
    [type]
        [description]
    """
    name_cond_py = "cond.py" 
    fldname = input("Please input the name of folder which contains regression shape files and calculating condition such as {}.\n>>".format(name_cond_py))
    if os.path.exists(fldname):
        if os.path.exists(os.path.join(fldname, name_cond_py)):
            sys.path.append(os.path.join(os.path.dirname(__file__), fldname))
            import cond
            dic_cond = {"PARAM": cond.PARAM,
                        "BOUND": cond.BOUND,
                        "CONTOUR_PLOT": cond.CONTOUR_PLOT
                        }
        else:
            print("There is no python module of calculation condition, \"{}\"".format(name_cond_py))
            sys.exit()
    else:
        print("There is no such a folder, \"{}\"".format(fldname))
        sys.exit()
    return fldname, dic_cond



if __name__ == "__main__":
# %%
    # PARAM = {"rho_f": 1190,     # [kg/m^3] solid fuel density
    #          "M_ox": 32.0e-3,   # [kg/mol]
    #          "T": 300,          # [K] oxidizer tempreature
    #          "Cr_init": 15e-6,# Initial guess value for coefficent fitting
    #          "z_init": 0.4,     # Initial guess value for coefficent fitting
    #          "m_init": -0.26,   # Initial guess value for coefficent fitting
    #          "C1": 1.39e-7,  # experimental constant of experimental regression rate formula
    #          "C2": 1.61e-9,  # experimental constant of experimental regression rate formula
    #          "n": 1.0,       # experimental exponent constant of pressure
    #          "dx": 0.1e-3,      # [m] space resolution
    #          "r_0": 0.0,        # r=0.0 when x = 0, boudanry condition
    #          "rdot_0": 0.0,     # rdot=0 when x = 0, boundary condition
    #          "Vf_mode": False , # mode selection using axial or radial integretioin for mf calculation
    #          "use_Vf": True    # mode selection for using experimental Vf insted of Vf empirical formula or not.
    #         }
    
    # BOUND = {"Cr": (1.0e-6, 30.0e-6),
    #          "z": (0.0, 1.0),
    #          "m": (-0.5, 0.0)
    #          }

    # CONTOUR_PLOT = {"plot": True,
    #                 "Cr_bnd": (14.0e-6, 19.0e-6),      # plot range of Cr
    #                 "z_bnd": (0.2, 0.5),               # plot range of z
    #                 "m_bnd": (-0.5, -0.2),             # plot range of m
    #                 "resol": 25,                      # the number of calculating point for x and y axis direction
    #                 "thirdparam": "Cr",                # select the thrid parameter. selected parameter is varied with the number of "num_fig"
    #                 "num_fig": 5                       # the number of figures. This value is the same of the number of variety of "thirdparam".
    #                 }
    
# %%
    ## read the condition file
    FLDNAME, COND = read_cond()
    PARAM = COND["PARAM"]
    BOUND = COND["BOUND"]
    CONTOUR_PLOT = COND["CONTOUR_PLOT"]
    
    ## generating instance
    inst = Fitting(PARAM, FLDNAME)

    ## optimization for model constants
    RES_TMP = inst.optimize_modelconst(mode="R2", method="global", bounds=[BOUND["Cr"], BOUND["z"], BOUND["m"]])
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
    # FLDNAME = datetime.now().strftime("%Y_%m%d_%H%M%S")
    # os.mkdir(FLDNAME)
    with open(os.path.join(FLDNAME, "result.json"), "w") as f:
        json.dump(OUTPUT, f, ensure_ascii=False, indent=4)
    
    # output figures which compare experimental and calculated result
    FLDNAME_EXPCOMP = "fig_expcomp"
    os.mkdir(os.path.join(FLDNAME, FLDNAME_EXPCOMP))
    FIG_COMP = inst.gen_excomp_figlist(RESULT["Cr"], RESULT["z"], RESULT["m"], mode="R2")
    for testname, dic in FIG_COMP.items():
        dic["fig"].savefig(os.path.join(FLDNAME, FLDNAME_EXPCOMP, "{}.png".format(testname)), dpi=300)

# %%
    ## output figures which is contour map for calculated coefficient
    if CONTOUR_PLOT["plot"]:
        print("\nStart generating the contour map of coefficient.")
        print("If you do not want a contourmap, please execute keyboard interruption")
        FLDNAME_CONTOUR = "fig_coeff_contour"
        os.mkdir(os.path.join(FLDNAME, FLDNAME_CONTOUR))
        FIG_COEFF = inst.gen_R_R2_figlist(bounds=[CONTOUR_PLOT["Cr_bnd"], CONTOUR_PLOT["z_bnd"], CONTOUR_PLOT["m_bnd"]],\
            mode="R2", resolution=CONTOUR_PLOT["resol"], thirdparam=CONTOUR_PLOT["thirdparam"], num_fig=CONTOUR_PLOT["num_fig"],\
            fldname=os.path.join(FLDNAME, FLDNAME_CONTOUR))
        for coef_dic in FIG_COEFF:
            coef_dic["fig"].savefig(os.path.join(FLDNAME, FLDNAME_CONTOUR,\
                "{0}={1:5.3e}.png".format(CONTOUR_PLOT["thirdparam"], coef_dic["third"])), dpi=300)
    else:
        pass
    print("Compleated!")