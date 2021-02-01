# -*- coding: utf-8 -*-
"""
Condition file for "main_analy.py" as a python module
"""

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
            "r_0": 0.0,        # r=0.0 when x = 0, boudanry condition
            "rdot_0": 0.0,     # rdot=0 when x = 0, boundary condition
            "Vf_mode": False , # mode selection using axial or radial integretioin for mf calculation
            "use_Vf": True    # mode selection for using experimental Vf insted of Vf empirical formula or not.
        }

BOUND = {"Cr": (1.0e-6, 30.0e-6),
            "z": (0.0, 1.0),
            "m": (-0.5, 0.0)
            }

CONTOUR_PLOT = {"plot": True,
                "Cr_bnd": (14.0e-6, 19.0e-6),      # plot range of Cr
                "z_bnd": (0.2, 0.5),               # plot range of z
                "m_bnd": (-0.5, -0.2),             # plot range of m
                "resol": 25,                      # the number of calculating point for x and y axis direction
                "thirdparam": "Cr",                # select the thrid parameter. selected parameter is varied with the number of "num_fig"
                "num_fig": 5                       # the number of figures. This value is the same of the number of variety of "thirdparam".
                }