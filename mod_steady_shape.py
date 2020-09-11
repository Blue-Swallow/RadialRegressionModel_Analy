
# coding: utf-8
"""
# Content: Fuel Regression Shape of EBHR after Steady State
# Author: Ayumu Tsuji @Hokkaido University

# Description:
Estimate the single port fuel regression shape of axial-injection end-burning
hybrid rocket when the combustion state becomes steady state.
To estimate the shape, this program needs some empirical constant,Cr; z; m, k,
which are obtained by experimental results.
"""

import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from scipy.integrate import simps
from scipy.integrate import trapz
from scipy.optimize import newton
from datetime import datetime
from tqdm import tqdm

mplstyle.use("tj_origin.mplstyle")



#%% 関数定義
class Main:
    def __init__(self, Pc, Vox, **cond):
        self.Pc = Pc
        self.Vox = Vox
        self.cond = cond
        self.x, self.r, self.rdot = self.initialize()

    def func_Vf(self, Vox, Pc):
        C1 = self.cond["C1"]
        C2 = self.cond["C2"]
        n = self.cond["n"]
        Vf = (C1/Vox + C2)*np.power(Pc, n)
        return(Vf)

    def func_mf(self, i, **kwargs):
        r_0 = self.cond["r_0"]
        rdot_0 = self.cond["rdot_0"]
        d = self.cond["d"]
        rho_f = self.cond["rho_f"]
        if i == 0:
            x_range = np.array([0.0])
            r_range = np.array([r_0])
            rdot_range = np.array([rdot_0])
        else:
            x_range = self.x[:i]
            r_range = self.r[:i]
            rdot_range = self.rdot[:i]
        if self.cond["Vf_mode"]:
            if self.cond["use_Vf"]:
                Vf = self.cond["Vf"]
            else:
                Vf = self.func_Vf(self.Vox, self.Pc)
            mf = rho_f*Vf*np.pi*(np.power(self.r[i], 2) +d*self.r[i] )  # calculate using integretion for radial direction
        else:
            mf = 2*np.pi*rho_f*simps((r_range + d/2)*rdot_range, x_range)  # calculate using integretion for axial direction
        return(mf)

    def func_mox(self, i):
        d = self.cond["d"]
        M_ox = self.cond["M_ox"]
        T = self.cond["T"]
        Rstr = 8.3144598 # [J/mol-K]
        rho_ox = self.Pc/((Rstr/M_ox)*T)
        mox = rho_ox*self.Vox*(np.pi*np.power(d,2)/4)
        return(mox)

    def func_G(self, i, ri):
        d = self.cond["d"]
        G = (self.func_mf(i) + self.func_mox(i)) / (np.pi*np.power(2*ri+d, 2)/4)
        return(G)

    def func_rdot(self, i, ri):
        G = self.func_G(i, ri)
        Cr = self.cond["Cr"]
        z = self.cond["z"]
        m = self.cond["m"]
        # k = self.cond["k"]
        rdot_0 = self.cond["rdot_0"]
        if self.x[i] == 0:
            rdot0 = rdot_0
        else:
            rdot0 = Cr*np.power(G, z)*np.power(self.x[i], m) # origin formula
        # # th = k*np.power(G, 0.8)/self.Pc       # equation obtained from dissertation written by Hashimoto
        # # rdoti = rdot0 *np.sqrt(2/th) *np.sqrt(1 -1/th*(1 -np.exp(-th)))
        rdoti = rdot0
        return(rdoti)

    def func_r(self, i, rdoti):
        rdot_0 = self.cond["rdot_0"]
        if i == 0:
            x_range = np.array([0.0])
            rdot_range = np.array([rdot_0])
        else:
            rdot_range = self.rdot[:i]
            rdot_range = np.append(rdot_range, rdoti)
            x_range = np.array(self.x[:i])
            x_range = np.append(x_range, self.x[i])
        if self.cond["use_Vf"]:
            ri = trapz(rdot_range, x_range)/self.cond["Vf"]
        else:
            ri = trapz(rdot_range, x_range)/self.func_Vf(self.Vox, self.Pc)
        return(ri)

    def error_rdot(self, rdoti, i):
        ri = self.func_r(i, rdoti)
        rdot_cal = self.func_rdot(i, ri)
        error = (rdoti - rdot_cal)/rdot_cal
        return(error)
    
    def iterat_rdot(self, i):
        r_0 = self.cond["r_0"]
        if i == 0:
            rdoti = self.func_rdot(i, r_0)
        else:
            rdoti = newton(self.error_rdot, x0=self.rdot[i-1] ,args=(i,))
        return(rdoti)

    def initialize(self):
        """Generate variable array
        
        Parameters
        ----------
        **kwargs: keyword variable
            **kwargs must include the follwoing key; "dx", "x_max", "r_0", "rdot_0"

        Returns
        -------
        x: 1d-ndarray of float
            position [m]
        r: 1d-ndarray of float
            initialized fuel regression distance array
        rdot 1d-ndarray of float
            initialized fuel regression rate array
        """
        dx = self.cond["dx"]
        x_max = self.cond["x_max"]
        r_0 = self.cond["r_0"]
        rdot_0 = self.cond["rdot_0"]
        x = np.arange(0.0, x_max+dx, dx)
        r = np.zeros(int(round((x_max+dx)/dx, 0)), float)
        rdot = np.zeros(int(round((x_max+dx)/dx)), float)
        r = np.array([r_0 for i in r])
        rdot = np.array([rdot_0 for i in rdot])
        return x, r, rdot

    def exe(self):
        self.x, self.r, self.rdot = self.initialize()
        for i in range(len(self.r)):
            self.rdot[i] = self.iterat_rdot(i)
            self.r[i] = self.func_r(i, self.rdot[i])
        return self.x, self.r, self.rdot

def plot_r(ax1, x, r, Pc, Vox, mf, **cond):
    x_max = cond["x_max"]
    y_max = cond["y_max_r"]
    # legend_text = r"$P_c$=" + str(round(Pc*1e-6,2)) + " MPa, "\
    #              + r"$V_{ox}$=" + str(round(Vox, 1)) + " m/s, "\
    #              + r"$\dot m_f$=" + str(round(mf*1e+6, 1)) + " mg/s"
    legend_text = r"$P_c$=" + str(round(Pc*1e-6,2)) + " MPa"
    ax1.plot(x*1.0e+3, r*1.0e+3, label=legend_text)
    ax1.set_xlabel("Axial distance $x$ [mm]")
    ax1.set_ylabel("Radial regression distance $r$ [mm]")
    # ax1.text(x_max*1.0e+3*0.8, y_max*1.0e+3*0.9, r"$P_c$={} MPa".format(Pc*1.0e-6))
    # ax1.text(x_max*1.0e+3*0.8, y_max*1.0e+3*0.8, r"$Vox$={} m/s".format(Vox))
    ax1.legend(fontsize=24)
    ax1.set_ylim(0,y_max*1.0e+3)
    # ax1.set_xlim(0,x_max*1.0e+3)
    ax1.set_xlim(0, 15.0)
    ax1.grid()
    return ax1
    # fig1.savefig(os.path.joint(fldname, "r_steady.png"))
    # fig1.show()

def plot_rdot(ax2, x, rdot, Pc, Vox, **cond):
    x_max = cond["x_max"]
    y_max = cond["y_max_rdot"]
    legend_text = r"$P_c$=" + str(round(Pc*1e-6,2)) + " MPa," + r"$V_{ox}$=" + str(round(Vox, 1)) + " m/s"
    ax2.plot(x*1.0e+3, rdot*1.0e+3, label=legend_text)
    ax2.set_xlabel("Axial distance $x$ [mm]")
    ax2.set_ylabel("Radial regression rate $\dot r$ [mm/s]")
    #plt.text(x_max*1.0e+3*0.8, 1.0, r"$P_c$={} MPa".format(Pc*1.0e-6))
    #plt.text(x_max*1.0e+3*0.8, 0.9, r"$Vox$={} m/s".format(Vox))
    ax2.legend()
    ax2.set_ylim(0, y_max*1.0e+3)
    ax2.set_xlim(0,x_max*1.0e+3)
    ax2.grid()
    return ax2
    # fig2.savefig(os.path.joint(fldname, "rdot_steady.png"))
    # fig2.show()


if __name__ == "__main__":
#%% 計算パラメータの定義
    PARAM = {"d": 0.3e-3,       # [m] port diameter
             "rho_f": 1190,     # [kg/m^3] solid fuel density
             "M_ox": 32.0e-3,   # [kg/mol]
             "T": 300,          # [K] oxidizer tempreature
            #  "Cr": 4.58e-6,     
             "Cr": 3.01e-6,
            #  "Cr": 20.0e-6,
             "z": 0.9,
            # "z": 0.6,
             "m": -0.2,
             "k": 3.0e+4,
             "C1": 1.39e-7,  # experimental constant of experimental regression rate formula
             "C2": 1.61e-9,  # experimental constant of experimental regression rate formula
             "n": 1.0,       # experimental exponent constant of pressure
             "dx": 0.1e-3,      # [m] space resolution
             "x_max": 30.0e-3,  # [m] maximum calculation region
             "y_max_r": 1.5e-3,   # [m] maximum plot width of radial regression distance
             "y_max_rdot": 2.0e-3,   # [m] maximum plot width of radial regression rate
             "r_0": 0.0,        # r=0.0 when x = 0, boudanry condition
             "rdot_0": 0.0,     # rdot=0 when x = 0, boundary condition
             "Vf_mode": False   # mode selection using axial or radial integretioin for mf calculation
            }

    Pc_range = np.arange(0.25e+6, 1.50e+6, 0.25e+6)
    Vox_range =  np.arange(20, 81, 10)
    Vox = 30.0 # [m/s] oxidizer port velocity
    Pc = 0.25e+6 # [MPa] chamber pressure
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    ax3 = fig3.add_subplot(111)
    ax4 = fig4.add_subplot(111)
    fldname = datetime.now().strftime("%Y_%m%d_%H%M%S") + "_SteadyShape"
    os.makedirs(fldname)
    with open(os.path.join(fldname, "cond.json"), "w") as f:
        json.dump(PARAM, f, ensure_ascii=False, indent=4)
    for Pc_tmp in tqdm(Pc_range):
        inst1 = Main(Pc_tmp, Vox, **PARAM)
        x, r, rdot = inst1.exe()
        mf = inst1.func_mf(inst1.x.size-1)
        ax1 = plot_r(ax1, x, r, Pc_tmp, Vox, mf, **PARAM)
        ax2 = plot_rdot(ax2, x, rdot, Pc_tmp, Vox, **PARAM)
    for Vox_tmp in tqdm(Vox_range):
        inst2 = Main(Pc, Vox_tmp, **PARAM)
        x, r, rdot = inst2.exe()
        mf = inst2.func_mf(inst2.x.size-1)
        ax3 = plot_r(ax3, x, r, Pc, Vox_tmp, mf, **PARAM)
        ax4 = plot_rdot(ax4, x, rdot, Pc, Vox_tmp, **PARAM)
    fig1.savefig(os.path.join(fldname, "r_steady_Pc.png"))
    fig2.savefig(os.path.join(fldname, "rdot_steady_Pc.png"))
    fig3.savefig(os.path.join(fldname, "r_steady_Vox.png"))
    fig4.savefig(os.path.join(fldname, "rdot_steady_Vox.png"))