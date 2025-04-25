import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
import scipy.linalg as linalg


class HPADesigner:
    def __init__(
        self,
        n_div=4,
        max_plys=10,
        level=0,
        AIRFOIL=False,
        WIRE=True,
        DIHEDRAL=False,
        PAYLOAD=False,
        FINE_MODE=False,
    ):
        self.n_div = n_div  # division number of wing span
        self.max_plys = max_plys
        self.level = level
        self.AIRFOIL = AIRFOIL
        self.WIRE = WIRE
        self.DIHEDRAL = DIHEDRAL
        self.PAYLOAD = PAYLOAD
        self.FINE_MODE = FINE_MODE
        self.UPDATED_TABLE = False

        # aerodynamic coefficients
        path_airfoil = os.path.join(os.path.dirname(__file__), "airfoil_info") + "/"
        # files = np.sort(os.listdir(path_airfoil))
        files = np.array(
            ["DAE11.xlsx", "DAE21.xlsx", "DAE31.xlsx", "DAE41.xlsx"]
        )  # use the above line if your own airfoils are used
        self.airfoils = np.array([os.path.splitext(file)[0] for file in files])
        cl_slope = []
        cl_zero = []
        cd0 = []
        t_ratio = []
        for i, (file, airfoil) in enumerate(zip(files, self.airfoils)):
            cd0_temp = pd.read_excel(path_airfoil + file, sheet_name="Cd", index_col=0)
            cl_temp = pd.read_excel(path_airfoil + file, sheet_name="Cl")
            geo_temp = pd.read_excel(path_airfoil + file, sheet_name="geometry")
            cl_slope.append(cl_temp.iloc[0, 0])
            cl_zero.append(cl_temp.iloc[0, 1])
            t_ratio.append(geo_temp.iloc[0, 0])
            cd0.append(cd0_temp.values)
            re = cd0_temp.index.astype(float).values
            aoa = cd0_temp.columns.astype(float).values
        index = np.arange(len(self.airfoils))
        cd0 = np.array(cd0)
        self.cls_interp = interp1d(index, cl_slope)
        self.cl0_interp = interp1d(index, cl_zero)
        self.t_interp = interp1d(index, t_ratio)
        self.cd0_interp = RegularGridInterpolator((index, re, aoa), cd0, method="linear")
        self.re_max, self.re_min, self.aoa_max, self.aoa_min = (
            re.max(),
            re.min(),
            aoa.max(),
            aoa.min(),
        )

        # parameters
        self.n_aero = max(200, self.n_div * 10) if self.FINE_MODE else max(50, self.n_div * 5)
        self.n_struc = 6 * self.n_aero + 1
        self.gravity = 9.8
        self.rho = (
            0.1174 * self.gravity
        )  # 30 degree Celsius for worst flight condition with low air density
        self.visc_mu = 1.615e-5 * self.rho  # 30 degree Celsius
        # design&weight
        self.max_power = 400  # [W]
        self.v_min = 7.3  # [m/s]
        self.load_factor = 1.5
        self.safty_factor = 2.0
        self.pilot_weight = 60.0  # [kg]
        self.water_weight = 4.0  # [kg]
        self.density_rib_skin = 0.195  # [kg/m^2]
        self.max_dihedral_angle_at_root = 4.0  # [deg]
        self.max_dihedral_angle_at_tip = 8.0  # [deg]
        self.drivetrain_efficiency = 0.85 * 0.95  # propeller*drivetrain
        # drag
        self.calibration_factor = 1.1
        self.cdS_body = 0.09  # [m^2]
        self.cd_tail = 0.009
        self.horisontal_tail_volume = 0.415
        self.horisontal_tail_arm = 5.3  # [m]
        self.vertical_tail_volume = 0.0137
        self.vertical_tail_arm = 6.1  # [m]
        # wire
        self.base_wire_area = 2e-6  # [m^2]
        self.wire_max_tension = 135.0 * self.gravity  # [N]
        self.wire_density = 0.017 / self.base_wire_area  # [kg/m^3]
        self.wire_cd = 1.0
        self.wire_joint_weight = 0.25  # [kg]
        self.z_wire = 1.85  # [m]
        self.wire_iteration = 5
        # CFRP
        self.ultimate_strain_CFRP = 0.0027
        self.allowable_strain = self.ultimate_strain_CFRP / (self.load_factor * self.safty_factor)
        self.E_L = 170e9  # [Pa]
        self.E_T = 6.0e9  # [Pa]
        self.G_LT = 3.3e9  # [Pa]
        self.nu_L = 0.28
        self.nu_T = 0.01
        self.t_ply = 0.125e-3  # [m]
        self.density_CFRP = 1.35e3  # [kg/m^3]
        self.coef_rear_spar = 1.2  # for weight (main-spar:1 + rear-spar:0.1 + rear-spar arm:0.1)
        self.delta_width = 2e-3  # [m]

        # design variables
        if self.level == 0:
            self.n_x = self.n_div * 3 + 4
        elif self.level == 1:
            self.n_x = self.n_div * 6 + 2
        elif self.level == 2:
            self.n_x = self.n_div * (5 + 2 * self.max_plys) + 2
        if self.WIRE:
            self.i_x_wire = self.n_x
            self.n_x += 1
        if self.DIHEDRAL:
            self.i_x_dihedral = self.n_x
            self.n_x += 1
        if self.PAYLOAD:
            self.i_x_payload = self.n_x
            self.n_x += 1
        if self.AIRFOIL:
            self.n_x += self.n_div + 1

        # boundary
        m = self.max_plys
        self.lbound = np.zeros(self.n_x)
        self.ubound = np.ones(self.n_x)
        self.lbound[0 : self.n_div] = 8 / self.n_div  # length of beams [m]
        self.ubound[0 : self.n_div] = 20 / self.n_div  # length of beams [m]
        self.lbound[self.n_div : self.n_div * 2 + 1] = 0.4  # chord length [m]
        self.ubound[self.n_div : self.n_div * 2 + 1] = 1.5  # chord length [m]
        self.lbound[self.n_div * 2 + 1 : self.n_div * 2 + 2] = (
            4.0  # angle of attack at wing root [deg]
        )
        self.ubound[self.n_div * 2 + 1 : self.n_div * 2 + 2] = (
            7.0  # angle of attack at wing root [deg]
        )
        self.lbound[self.n_div * 2 + 2 : self.n_div * 3 + 2] = -8.0 / self.n_div  # washout [deg]
        self.ubound[self.n_div * 2 + 2 : self.n_div * 3 + 2] = 0.0  # washout [deg]
        if self.level == 0:
            self.lbound[self.n_div * 3 + 2 : self.n_div * 3 + 4] = (
                0.01  # reduction factor of allowable strain for deflection constraint handling
            )
            self.ubound[self.n_div * 3 + 2 : self.n_div * 3 + 4] = (
                1.0  # reduction factor of allowable strain for deflection constraint handling
            )
        if self.level > 0:
            self.lbound[self.n_div * 3 + 2 : self.n_div * 4 + 2] = (
                0.8  # beam diameter over max diameter of airfoil
            )
            self.ubound[self.n_div * 3 + 2 : self.n_div * 4 + 2] = (
                1.0  # beam diameter over max diameter of airfoil
            )
        if self.level == 1:
            self.lbound[self.n_div * 4 + 2 : self.n_div * 6 + 2] = (
                0.01  # reduction factor of allowable strain for deflection constraint handling
            )
            self.ubound[self.n_div * 4 + 2 : self.n_div * 6 + 2] = (
                1.0  # reduction factor of allowable strain for deflection constraint handling
            )
        elif self.level == 2:
            self.lbound[self.n_div * 4 + 2 : self.n_div * (4 + m) + 2] = 0.0  # length of UD ply
            self.ubound[self.n_div * 4 + 2 : self.n_div * (4 + m) + 2] = 1.0  # length of UD ply
            self.lbound[self.n_div * (4 + m) + 2 : self.n_div * (4 + m * 2) + 2] = (
                self.delta_width
            )  # delta width of UD ply [m]
            self.ubound[self.n_div * (4 + m) + 2 : self.n_div * (4 + m * 2) + 2] = (
                260e-3  # delta width of UD ply [m] =pi*max radius=pi*1.5*0.11/2=0.0825pi=0.259
            )
            self.lbound[self.n_div * (4 + m * 2) + 2 : self.n_div * (5 + m * 2) + 2] = (
                0.0  # fixed position
            )
            self.ubound[self.n_div * (4 + m * 2) + 2 : self.n_div * (5 + m * 2) + 2] = (
                1.0  # fixed position
            )
        if self.WIRE:
            self.lbound[self.i_x_wire] = 0.0  # wire tension [kg]
            self.ubound[self.i_x_wire] = self.wire_max_tension  # wire tension [kg]
        if self.DIHEDRAL:
            self.lbound[self.i_x_dihedral] = 0.0  # dihedral angle [deg]
            self.ubound[self.i_x_dihedral] = (
                self.max_dihedral_angle_at_root
            )  # dihedral angle [deg]
        if self.PAYLOAD:
            self.lbound[self.i_x_payload] = 0.0  # payload [kg]
            self.ubound[self.i_x_payload] = self.pilot_weight  # payload [kg]
        if self.AIRFOIL:
            self.lbound[-self.n_div - 1 :] = 0  # airfoil
            self.ubound[-self.n_div - 1 :] = len(self.airfoils) - 1  # airfoil

        # airfoils for level=0
        self.airfoil_baseline = np.full(self.n_div + 1, np.argmax(self.airfoils == "DAE31"))
        if self.n_div < 3:
            self.airfoil_baseline[0 : self.n_div] = np.argmax(self.airfoils == "DAE21")
        elif self.n_div == 3:
            self.airfoil_baseline[0:2] = np.argmax(self.airfoils == "DAE11")
            self.airfoil_baseline[2] = np.argmax(self.airfoils == "DAE21")
        elif self.n_div > 3:
            self.airfoil_baseline[0:2] = np.argmax(self.airfoils == "DAE11")
            self.airfoil_baseline[2] = np.argmax(self.airfoils == "DAE21")
            self.airfoil_baseline[-1] = np.argmax(self.airfoils == "DAE41")

    def _x2param(self, x, NORMALIZED=False):
        def sigmoid(x, a=50.0, b=0.5):
            # Exact 0 and 1 are needed at wing root and tip in structural analysis
            # 4-digit means that machining accuracy of CFRP pipes will be "beam length"*1e-4 [m]
            return np.round(1.0 / (1.0 + np.exp(-a * (x - b))), 4)

        if NORMALIZED:
            x = self.lbound + (self.ubound - self.lbound) * x
        span = x[0 : self.n_div].sum() * 2
        y_div = np.hstack([0.0, x[0 : self.n_div]]).cumsum() / (span * 0.5)
        y_div = np.where(y_div < 1.0, y_div, 1.0)
        y_chord = x[self.n_div : self.n_div * 2 + 1]
        y_aoa = x[self.n_div * 2 + 1 : self.n_div * 3 + 2].cumsum()
        if self.AIRFOIL:
            y_airfoil = x[-self.n_div - 1 :]
        else:
            y_airfoil = self.airfoil_baseline.copy()

        if self.n_div > 1:
            y_wire = 0.95 * y_div[int(self.n_div / 2)]
        else:
            y_wire = 0.5 * y_div[1]
        if self.WIRE:
            wire_tension = x[self.i_x_wire]
        else:
            wire_tension = 0.0
        if self.DIHEDRAL:
            dihedral_angle_at_root = x[self.i_x_dihedral]
        else:
            dihedral_angle_at_root = self.max_dihedral_angle_at_root
        if self.PAYLOAD:
            payload = x[self.i_x_payload]
        else:
            payload = 0.0
        strain_ratios = np.ones([2, self.n_div])

        ply_wing = []
        if self.level == 0:
            y_diameter = np.array(
                [
                    (
                        np.linspace(y_chord[i - 1], y_chord[i], 100)
                        * self.t_interp(np.linspace(y_airfoil[i - 1], y_airfoil[i], 100))
                    ).min()
                    for i in range(1, len(y_div))
                ]
            )
            y_diameter = np.hstack([y_diameter[0], y_diameter])
            phis = 30 + np.rad2deg(
                np.tile(self.delta_width * np.arange(self.max_plys), [self.n_div, 1])
                / (0.5 * np.tile(y_diameter[1:], [self.max_plys, 1]).T)
            )
            phis = np.clip(phis, 0, 179)
            for i in range(self.n_div):
                if i == int(self.n_div / 2):
                    plys = np.array(
                        [
                            [90, 180, 0, 1, self.t_ply],
                            [30, 180, 0, 1, self.t_ply],
                            [-30, 180, 0, 1, self.t_ply],
                            [90, 180, 0, 1, self.t_ply],
                        ]
                    )
                else:
                    plys = np.array(
                        [
                            [90, 180, 0, 1, self.t_ply],
                            [45, 180, 0, 1, self.t_ply],
                            [-45, 180, 0, 1, self.t_ply],
                            [90, 180, 0, 1, self.t_ply],
                        ]
                    )
                for j in range(self.max_plys):
                    plys = np.vstack([plys, [0, phis[i, j], 0, 1, self.t_ply]])
                ply_wing.append(plys)
            strain_ratios = np.tile(x[self.n_div * 3 + 2 : self.n_div * 3 + 4], [self.n_div, 1]).T

        elif self.level > 0:
            y_diameter = x[self.n_div * 3 + 2 : self.n_div * 4 + 2] * np.array(
                [
                    (
                        np.linspace(y_chord[i - 1], y_chord[i], 100)
                        * self.t_interp(np.linspace(y_airfoil[i - 1], y_airfoil[i], 100))
                    ).min()
                    for i in range(1, len(y_div))
                ]
            )
            y_diameter = np.hstack([y_diameter[0], y_diameter])
            if self.level == 1:
                phis = 30 + np.rad2deg(
                    np.tile(self.delta_width * np.arange(self.max_plys), [self.n_div, 1])
                    / (0.5 * np.tile(y_diameter[1:], [self.max_plys, 1]).T)
                )
                phis = np.clip(phis, 0, 179)
                for i in range(self.n_div):
                    if i == int(self.n_div / 2):
                        plys = np.array(
                            [
                                [90, 180, 0, 1, self.t_ply],
                                [30, 180, 0, 1, self.t_ply],
                                [-30, 180, 0, 1, self.t_ply],
                                [90, 180, 0, 1, self.t_ply],
                            ]
                        )
                    else:
                        plys = np.array(
                            [
                                [90, 180, 0, 1, self.t_ply],
                                [45, 180, 0, 1, self.t_ply],
                                [-45, 180, 0, 1, self.t_ply],
                                [90, 180, 0, 1, self.t_ply],
                            ]
                        )
                    for j in range(self.max_plys):
                        plys = np.vstack([plys, [0, phis[i, j], 0, 1, self.t_ply]])
                    ply_wing.append(plys)
                strain_ratios = x[self.n_div * 4 + 2 : self.n_div * 6 + 2].reshape(2, -1)
            elif self.level == 2:
                length = x[self.n_div * 4 + 2 : self.n_div * (4 + self.max_plys) + 2].reshape(
                    self.n_div, self.max_plys
                )
                length = 1 - np.clip(length.cumsum(axis=1), 0, 1)
                fixed = x[
                    self.n_div * (4 + self.max_plys * 2) + 2 : self.n_div * (5 + self.max_plys * 2)
                    + 2
                ].reshape(self.n_div, 1)
                fixed = sigmoid(fixed)
                tip = fixed + length * (1 - fixed)
                root = fixed * (1 - length)
                phis = x[
                    self.n_div * (4 + self.max_plys) + 2 : self.n_div * (4 + self.max_plys * 2) + 2
                ].reshape(self.n_div, self.max_plys)
                phis[:, 0] -= self.delta_width
                phis = 30 + np.rad2deg(
                    phis / (0.5 * np.tile(y_diameter[1:], [self.max_plys, 1]).T)
                ).cumsum(axis=1)
                phis = np.clip(phis, 0, 179)
                for i in range(self.n_div):
                    if i == int(self.n_div / 2):
                        plys = np.array(
                            [
                                [90, 180, 0, 1, self.t_ply],
                                [30, 180, 0, 1, self.t_ply],
                                [-30, 180, 0, 1, self.t_ply],
                                [90, 180, 0, 1, self.t_ply],
                            ]
                        )
                    else:
                        plys = np.array(
                            [
                                [90, 180, 0, 1, self.t_ply],
                                [45, 180, 0, 1, self.t_ply],
                                [-45, 180, 0, 1, self.t_ply],
                                [90, 180, 0, 1, self.t_ply],
                            ]
                        )
                    for j in range(self.max_plys):
                        plys = np.vstack(
                            [plys, [0, phis[i, j], root[i, j], tip[i, j], self.t_ply]]
                        )
                    ply_wing.append(plys)

        return (
            span,
            y_div,
            y_chord,
            y_aoa,
            y_diameter,
            y_airfoil,
            y_wire,
            wire_tension,
            dihedral_angle_at_root,
            payload,
            ply_wing,
            strain_ratios,
        )

    def _param2x(
        self,
        span,
        y_div,
        y_chord,
        y_aoa,
        y_diameter,
        y_airfoil,
        y_wire=0,
        wire_tension=0,
        dihedral_angle_at_root=4.0,
        payload=0,
        ply_wing=[],
        strain_ratios=[],
    ):
        def logit(x, a=50.0, b=0.5):
            if x <= 0.0:
                y = 0.0
            elif x >= 1.0:
                y = 1.0
            else:
                y = b - 1 / a * np.log((1 - x) / x)
            return y

        if len(y_airfoil) == self.n_div:
            y_airfoil = np.hstack([y_airfoil, y_airfoil[-1]])
        if len(y_diameter) == self.n_div:
            y_diameter = np.hstack([y_diameter[0], y_diameter])
        x = np.zeros(self.n_x)
        x[: self.n_div] = np.diff(y_div) * 0.5 * span
        x[self.n_div : self.n_div * 2 + 1] = y_chord
        x[self.n_div * 2 + 1 : self.n_div * 3 + 2] = np.hstack([y_aoa[0], np.diff(y_aoa)])
        if self.AIRFOIL:
            x[-self.n_div - 1 :] = y_airfoil
        if self.WIRE:
            x[self.i_x_wire] = wire_tension
        if self.DIHEDRAL:
            x[self.i_x_dihedral] = dihedral_angle_at_root
        if self.PAYLOAD:
            x[self.i_x_payload] = payload

        if self.level == 0:
            if len(strain_ratios) > 0:
                x[self.n_div * 3 + 2 : self.n_div * 3 + 4] = strain_ratios.min(axis=1)
        elif self.level > 0:
            x[self.n_div * 3 + 2 : self.n_div * 4 + 2] = y_diameter[1:] / np.array(
                [
                    (
                        np.linspace(y_chord[i - 1], y_chord[i], 100)
                        * self.t_interp(np.linspace(y_airfoil[i - 1], y_airfoil[i], 100))
                    ).min()
                    for i in range(1, len(y_div))
                ]
            )
            if self.level == 1:
                if len(strain_ratios) > 0:
                    x[self.n_div * 4 + 2 : self.n_div * 6 + 2] = strain_ratios.reshape(-1)
                else:
                    x[self.n_div * 4 + 2 : self.n_div * 6 + 2] = 1.0
            elif self.level == 2:
                for i, plys in enumerate(ply_wing):
                    n_entire = np.sum(plys[:, 1] >= 180)
                    root = plys[n_entire:, 2]
                    tip = plys[n_entire:, 3]
                    length = tip - root
                    ifix = np.argmin(np.where(length > 0, length, 1))
                    if (1 - tip[ifix] + root[ifix]) == 0:
                        fixed = 0.0
                    else:
                        fixed = root[ifix] / (1 - tip[ifix] + root[ifix])
                    length = 1 - length
                    x[
                        self.n_div * 4 + 2 + self.max_plys * i : self.n_div * 4
                        + 2
                        + self.max_plys * i
                        + len(length)
                    ] = np.hstack([length[0], np.diff(length)])
                    phis = (
                        np.deg2rad(np.diff(np.hstack([30, plys[n_entire:, 1]])))
                        * 0.5
                        * y_diameter[i + 1]
                    )
                    phis[0] += self.delta_width
                    x[
                        self.n_div * (4 + self.max_plys) + 2 + self.max_plys * i : self.n_div
                        * (4 + self.max_plys)
                        + 2
                        + self.max_plys * i
                        + len(length)
                    ] = phis
                    x[
                        self.n_div * 4 + 2 + self.max_plys * i + len(length) : self.n_div * 4
                        + 2
                        + self.max_plys * (i + 1)
                    ] = 1.0
                    x[self.n_div * (4 + self.max_plys * 2) + 2 + i] = logit(fixed)
        return x

    def baseline(self):
        span = 32  # [m]
        y_div = np.array([0, 4.6, 8.6, 12.8, 16.0]) / (32 * 0.5)
        y_chord = np.array([1.05, 1.05, 0.903, 0.7455, 0.462])  # [m]
        y_aoa = np.array([4.8, 4.8, 4.8, 3.2, 2.0])  # [deg]
        y_airfoil_name = np.array(["DAE21", "DAE21", "DAE31", "DAE31", "DAE31"])
        y_airfoil = np.array([np.argmax(self.airfoils == name) for name in y_airfoil_name])
        y_diameter = np.array(
            [
                (
                    np.linspace(y_chord[i - 1], y_chord[i], 100)
                    * self.t_interp(np.linspace(y_airfoil[i - 1], y_airfoil[i], 100))
                ).min()
                for i in range(1, len(y_div))
            ]
        )
        if self.n_div > 1:
            y_wire = 0.95 * y_div[int(self.n_div / 2)]  # 8.2/16
        else:
            y_wire = 0.5 * y_div[1]
        wire_tension = 135 * self.gravity
        dihedral_angle_at_root = 4.0  # [deg]
        payload = 0.0  # [kg]
        n_plys_partial = np.array([8, 8, 8, 6])
        plys1 = np.array(
            [
                [90, 180, 0, 1, self.t_ply],
                [45, 180, 0, 1, self.t_ply],
                [-45, 180, 0, 1, self.t_ply],
                [90, 180, 0, 1, self.t_ply],
                [0, 30, 0, 1, self.t_ply],
                [0, 35, 0, 1.0, self.t_ply],
                [0, 40, 0, 1.0, self.t_ply],
                [0, 45, 0, 1.0, self.t_ply],
                [0, 50, 0, 1.0, self.t_ply],
                [0, 55, 0, 3.6 / 4.6, self.t_ply],
                [0, 65, 0, 2.6 / 4.6, self.t_ply],
                [0, 75, 0, 1.5 / 4.6, self.t_ply],
            ]
        )
        plys2 = np.array(
            [
                [90, 180, 0, 1, self.t_ply],
                [45, 180, 0, 1, self.t_ply],
                [-45, 180, 0, 1, self.t_ply],
                [90, 180, 0, 1, self.t_ply],
                [0, 35, 0, 1, self.t_ply],
                [0, 50, 0, 1.0, self.t_ply],
                [0, 60, 0, 1.0, self.t_ply],
                [0, 65, 0, 1.0, self.t_ply],
                [0, 70, 0, 1.0, self.t_ply],
                [0, 75, 0, 1.0, self.t_ply],
                [0, 80, 0.5 / 4, 1, self.t_ply],
                [0, 85, 2.3 / 4, 1, self.t_ply],
            ]
        )
        plys3 = np.array(
            [
                [90, 180, 0, 1, self.t_ply],
                [30, 180, 0, 1, self.t_ply],
                [-30, 180, 0, 1, self.t_ply],
                [90, 180, 0, 1, self.t_ply],
                [0, 30, 0, 1, self.t_ply],
                [0, 45, 0, 3.3 / 4.2, self.t_ply],
                [0, 55, 0, 2.9 / 4.2, self.t_ply],
                [0, 70, 0, 2.3 / 4.2, self.t_ply],
                [0, 80, 0, 1.8 / 4.2, self.t_ply],
                [0, 85, 0, 1.3 / 4.2, self.t_ply],
                [0, 90, 0, 0.85 / 4.2, self.t_ply],
                [0, 95, 0, 0.4 / 4.2, self.t_ply],
            ]
        )
        plys4 = np.array(
            [
                [90, 180, 0, 1, self.t_ply],
                [45, 180, 0, 1, self.t_ply],
                [-45, 180, 0, 1, self.t_ply],
                [90, 180, 0, 1, self.t_ply],
                [0, 35, 0, 1, self.t_ply],
                [0, 45, 0, 2.15 / 3.2, self.t_ply],
                [0, 55, 0, 1.8 / 3.2, self.t_ply],
                [0, 65, 0, 1.4 / 3.2, self.t_ply],
                [0, 75, 0, 0.95 / 3.2, self.t_ply],
                [0, 85, 0, 0.45 / 3.2, self.t_ply],
            ]
        )
        ply_wing = [plys1, plys2, plys3, plys4]
        if self.level < 2:
            strain_ratios = np.array([[0.88, 0.88, 0.88, 0.88], [0.66, 0.66, 0.66, 0.66]])
        else:
            strain_ratios = np.ones([2, self.n_div])
        return self._param2x(
            span,
            y_div,
            y_chord,
            y_aoa,
            y_diameter,
            y_airfoil,
            y_wire,
            wire_tension,
            dihedral_angle_at_root,
            payload,
            ply_wing,
            strain_ratios,
        )

    def _evaluate(self):
        self.UPDATED_TABLE = False
        # make data table
        self.aero = self._interpolate()
        # lifting-line theory
        self._lifting_line()
        # refine grid for structural analysis
        self.wing = self._refine_grid(self.aero)
        # wing weight
        self._compute_wing_weight()
        # velocity&lift distribution
        self._compute_local_lift()
        # bending stiffness
        self._stiffness()
        # zero-lift load & deflection
        self._evaluate_zerolift_deflection()
        # flight load & deflection
        self._evaluate_flight_deflection()
        # twist by AoA change
        self._evaluate_twist()
        # inner optimization for ply
        if self.level < 2:
            self._optimize_ply()
        # constraint functions
        self._compute_constraints()
        # performance
        if self.FINE_MODE:
            self._compute_power()
            self._update_wing_table()
            self.UPDATED_TABLE = True
        else:
            self._compute_power_aero()

    def evaluate_performance(
        self,
        span,
        y_div,
        y_chord,
        y_aoa,
        y_diameter,
        y_airfoil,
        y_wire,
        wire_tension,
        dihedral_angle_at_root,
        payload,
        n_plys_partial=[],
        strain_ratios=[],
        ply_wing=[],
    ):
        if len(y_airfoil) == self.n_div:
            y_airfoil = np.hstack([y_airfoil, y_airfoil[-1]])
        if len(y_diameter) == self.n_div:
            y_diameter = np.hstack([y_diameter[0], y_diameter])
        self.x = self._param2x(
            span,
            y_div,
            y_chord,
            y_aoa,
            y_diameter,
            y_airfoil,
            y_wire,
            wire_tension,
            dihedral_angle_at_root,
            payload,
            ply_wing,
            strain_ratios,
        )
        (
            self.span,
            self.y_div,
            self.y_chord,
            self.y_aoa,
            self.y_diameter,
            self.y_airfoil,
            self.y_wire,
            self.wire_tension,
            self.dihedral_angle_at_root,
            self.payload,
        ) = (
            span,
            y_div,
            y_chord,
            y_aoa,
            y_diameter,
            y_airfoil,
            y_wire,
            wire_tension,
            dihedral_angle_at_root,
            payload,
        )

        if len(ply_wing) > 0:
            self.ply_wing = ply_wing
        else:
            if len(n_plys_partial) == 0:
                n_plys_partial = np.full(self.n_div, self.max_plys)
            phis = 30 + np.rad2deg(
                np.tile(self.delta_width * np.arange(self.max_plys), [self.n_div, 1])
                / (0.5 * np.tile(self.y_diameter[1:], [self.max_plys, 1]).T)
            )
            phis = np.clip(phis, 0, 179)
            self.ply_wing = []
            for i in range(self.n_div):
                n_ply = int(np.round(n_plys_partial[i]))
                if i == int(self.n_div / 2):
                    plys = np.array(
                        [
                            [90, 180, 0, 1, self.t_ply],
                            [30, 180, 0, 1, self.t_ply],
                            [-30, 180, 0, 1, self.t_ply],
                            [90, 180, 0, 1, self.t_ply],
                        ]
                    )
                else:
                    plys = np.array(
                        [
                            [90, 180, 0, 1, self.t_ply],
                            [45, 180, 0, 1, self.t_ply],
                            [-45, 180, 0, 1, self.t_ply],
                            [90, 180, 0, 1, self.t_ply],
                        ]
                    )
                for j in range(n_ply):
                    plys = np.vstack([plys, [0, phis[i, j], 0, 1, self.t_ply]])
                self.ply_wing.append(plys)
        if len(strain_ratios) > 0:
            self.strain_ratios = strain_ratios
        else:
            self.strain_ratios = np.ones(self.n_div)
        self._evaluate()

    def evaluate_performance_from_x(self, x, NORMALIZED=False):
        (
            self.span,
            self.y_div,
            self.y_chord,
            self.y_aoa,
            self.y_diameter,
            self.y_airfoil,
            self.y_wire,
            self.wire_tension,
            self.dihedral_angle_at_root,
            self.payload,
            self.ply_wing,
            self.strain_ratios,
        ) = self._x2param(x, NORMALIZED)
        if NORMALIZED:
            self.x = self.lbound + (self.ubound - self.lbound) * x
        else:
            self.x = x
        self._evaluate()

    def _interpolate(self):
        section = pd.DataFrame(
            np.vstack(
                [np.arccos(self.y_div), self.y_chord, self.y_aoa, self.y_diameter, self.y_airfoil]
            ).T,
            columns=["theta", "chord", "aoa", "diameter", "airfoil"],
            index=self.y_div,
        )
        section["source"] = 0
        theta = 0.5 * np.pi * np.linspace(0, 1, self.n_aero + 1)
        aero = pd.DataFrame(
            np.full([len(theta), 6], np.nan),
            columns=["theta", "chord", "aoa", "diameter", "airfoil", "source"],
            index=np.cos(theta),
        )
        aero["theta"] = theta
        aero["source"] = 1
        aero = pd.concat([section, aero]).sort_index()
        aero.loc[:, "diameter"] = aero.loc[:, "diameter"].bfill()
        aero = aero.interpolate("index").ffill()
        aero = aero[aero["source"] == 1].drop(columns="source")
        return aero

    def _lifting_line(self):
        self.y0_aero = self.aero.index.values
        self.chord_aero = self.aero["chord"].values
        self.aoa_aero = self.aero["aoa"].values
        self.airfoil_aero = self.aero["airfoil"].values
        cl_slope = self.cls_interp(self.airfoil_aero)
        theta = self.aero["theta"].values[:-1]
        m = np.arange(1, 2 * self.n_aero, 2)
        ms = np.tile(m, [self.n_aero, 1])
        mu = cl_slope[:-1] * self.chord_aero[:-1] * 0.25 / self.span
        mus = np.tile(mu, [self.n_aero, 1]).T
        thetas = np.tile(theta, [self.n_aero, 1]).T
        absolute_aoa = np.deg2rad(self.aoa_aero - self.cl0_interp(self.airfoil_aero))
        self.aero["absolute aoa"] = np.rad2deg(absolute_aoa)
        A = (ms * mus + np.sin(thetas)) * np.sin(ms * thetas)
        b = mu * absolute_aoa[:-1] * np.sin(theta)
        xa = linalg.lu_solve(linalg.lu_factor(A), b)
        # lift and induced drag
        self.wing_area = (
            np.sum((self.y_chord[:-1] + self.y_chord[1:]) * np.diff(self.y_div) * 0.5) * self.span
        )
        self.aspect_ratio = self.span**2 / self.wing_area
        delta = np.sum(m[1:] * ((xa[1:] / xa[0]) ** 2))
        self.wing_efficiency = 1 / (1 + delta)
        self.CL = np.pi * self.aspect_ratio * xa[0]
        self.CDi = self.CL**2 / (np.pi * self.wing_efficiency * self.aspect_ratio)
        induced_aoa = np.sum(
            ms * np.tile(xa, [self.n_aero, 1]) * np.sin(ms * thetas) / np.sin(thetas), axis=1
        )
        induced_aoa = np.hstack([induced_aoa, absolute_aoa[-1]])
        self.aero["induced aoa"] = np.rad2deg(induced_aoa)
        self.aero["local cl"] = cl_slope * (absolute_aoa - induced_aoa)

    def _refine_grid(self, aero):
        y = np.linspace(0, 1, self.n_struc)
        fine_grid = pd.DataFrame(
            np.full([len(y), len(aero.columns)], np.nan), columns=aero.columns, index=y
        )
        fine_grid["source"] = 1
        aero["source"] = 0
        wing = pd.concat([aero, fine_grid]).sort_index().interpolate("index").ffill().bfill()
        wing = wing[wing["source"] == 1].drop(columns="source")
        diameter = np.full(len(wing), self.y_diameter[-1])
        for i in range(len(self.y_div) - 1, 0, -1):
            diameter[wing.index.values < self.y_div[i]] = self.y_diameter[i]
        wing["diameter"] = diameter
        self.y0 = wing.index.values
        self.y = self.y0 * self.span * 0.5
        wing.insert(0, "span position", self.y)
        self.chord = wing["chord"].values
        self.aoa = wing["aoa"].values
        self.diameter = wing["diameter"].values
        self.airfoil = wing["airfoil"].values
        self.local_cl = wing["local cl"].values
        self.dihedral_position = (
            0.5 * self.span * self.y0 * np.sin(np.deg2rad(self.dihedral_angle_at_root))
        )
        return wing

    def _compute_wing_weight(self):
        # beam
        self.beam_weight = np.zeros(len(self.ply_wing))
        self.wing_weight = np.zeros(self.n_struc)
        for i, plys in enumerate(self.ply_wing):
            y0_start = self.y_div[i]
            y0_end = self.y_div[i + 1]
            for ply in plys:
                phi = np.deg2rad(ply[1])
                R = np.where(
                    (self.y0 >= ply[2] * (y0_end - y0_start) + y0_start)
                    & (self.y0 <= ply[3] * (y0_end - y0_start) + y0_start),
                    0.5 * self.diameter,
                    0,
                )
                L = np.where(
                    (self.y0 >= ply[2] * (y0_end - y0_start) + y0_start)
                    & (self.y0 <= ply[3] * (y0_end - y0_start) + y0_start),
                    np.hstack([np.diff(self.y0), 0]),
                    0,
                )
                ply_weight = (
                    self.density_CFRP
                    * 2
                    * phi
                    * R.max()
                    * ply[4]
                    * 0.5
                    * self.span
                    * (ply[3] - ply[2])
                    * (y0_end - y0_start)
                )
                self.beam_weight[i] += ply_weight
                self.wing_weight += (
                    self.density_CFRP
                    * 2
                    * phi
                    * R
                    * ply[4]
                    * 0.5
                    * self.span
                    * L
                    * self.coef_rear_spar
                )
        # joint
        for y_joint in self.y_div[1:-1]:
            i = np.argmax(np.where(self.y0 < y_joint, self.y0, 0))
            self.wing_weight[i] += (
                0.5 * self.diameter[i + 1] * 2.293 + 3.373e-2
            )  # empirical estimation
        # rib and surface
        self.wing_weight += np.hstack(
            [
                self.density_rib_skin
                * np.diff(self.y0)
                * 0.5
                * self.span
                * 0.5
                * (self.chord[1:] + self.chord[:-1]),
                0,
            ]
        )
        # wire: simplified estimation for deciding v_inf and drag; actual wire weight and length must be computed for deflected wing
        i_wire = np.argmin(np.abs(self.y0 - self.y_wire))
        wire_ratio = self.wire_tension / self.wire_max_tension
        wire_area = self.base_wire_area * wire_ratio
        self.wire_diameter = 2 * np.sqrt(wire_area / np.pi)
        self.wire_length = np.sqrt((self.y_wire * self.span * 0.5) ** 2 + self.z_wire**2)
        wire_weight = self.wire_length * wire_area * self.wire_density + self.wire_joint_weight * (
            wire_ratio**1.5
        )
        self.wing_weight[i_wire] += wire_weight

    def _compute_local_lift(self):
        # weight & velocity
        self.body_tail_weight = (
            0.007 * (self.span - 15.0) ** 2.0 + 14.0 if self.span > 15.0 else 14.0
        )  # empirical estimation
        self.empty_weight = self.wing_weight.sum() * 2 + self.body_tail_weight
        self.weight = self.pilot_weight + self.water_weight + self.empty_weight + self.payload
        self.v_inf = np.sqrt(
            2 * self.weight * self.gravity / (self.rho * self.wing_area * self.CL)
        )
        self.local_lift = np.hstack(
            [
                0.5
                * self.rho
                * (self.v_inf**2)
                * (np.diff(self.y0) * self.span * 0.5)
                * 0.5
                * (self.local_cl[1:] * self.chord[1:] + self.local_cl[:-1] * self.chord[:-1]),
                0.0,
            ]
        )  # [N]

    def _compute_power_aero(self):
        # computation with aerodynamic (coarse) mesh points
        # drag & power
        # wing
        self.re_aero = self.chord_aero * self.v_inf * self.rho / self.visc_mu
        self.cd0_aero = self.cd0_interp(
            np.vstack(
                [
                    self.airfoil_aero,
                    np.clip(self.re_aero, self.re_min, self.re_max),
                    np.clip(self.aoa_aero, self.aoa_min, self.aoa_max),
                ]
            ).T
        )
        self.CD0 = (
            np.sum(
                0.5
                * (
                    self.cd0_aero[1:] * self.chord_aero[1:]
                    + self.cd0_aero[:-1] * self.chord_aero[1:]
                )
                * np.abs(np.diff(self.y0_aero))
                * self.span
                * 0.5
            )
            * 2
            / self.wing_area
        )
        # tail
        self.mac = (
            2.0
            / self.wing_area
            * np.sum(
                (np.abs(np.diff(self.y0_aero)) * self.span * 0.5)
                * 0.5
                * (self.chord_aero[1:] ** 2 + self.chord_aero[:-1] ** 2)
            )
        )
        self.holizontal_tail_area = (
            self.wing_area * self.mac * self.horisontal_tail_volume / self.horisontal_tail_arm
        )
        self.vertical_tail_area = (
            self.wing_area * self.span * self.vertical_tail_volume / self.vertical_tail_arm
        )
        # aircraft
        self.drag = (
            (
                self.calibration_factor
                * (
                    self.cdS_body
                    + self.CD0 * self.wing_area
                    + self.cd_tail * (self.holizontal_tail_area + self.vertical_tail_area)
                    + self.wire_cd * self.wire_length * self.wire_diameter
                )
                + self.CDi * self.wing_area
            )
            * 0.5
            * self.rho
            * self.v_inf**2
        )
        self.CD = self.drag / (0.5 * self.rho * self.wing_area * self.v_inf**2)
        # power
        self.power = self.drag * self.v_inf / self.drivetrain_efficiency
        self.aero["re"] = self.re_aero
        self.aero["cd0"] = self.cd0_aero
        self.power_constraint = self.power - self.max_power

    def _compute_power(self):
        # computation with structural (fine) mesh points
        # drag & power
        # wing
        self.re = self.chord * self.v_inf * self.rho / self.visc_mu
        self.cd0 = self.cd0_interp(
            np.vstack(
                [
                    self.airfoil,
                    np.clip(self.re, self.re_min, self.re_max),
                    np.clip(self.aoa, self.aoa_min, self.aoa_max),
                ]
            ).T
        )
        self.CD0 = (
            np.sum(
                0.5
                * (self.cd0[1:] * self.chord[1:] + self.cd0[:-1] * self.chord[1:])
                * np.abs(np.diff(self.y0))
                * self.span
                * 0.5
            )
            * 2
            / self.wing_area
        )
        # tail
        self.mac = (
            2.0
            / self.wing_area
            * np.sum(
                (np.abs(np.diff(self.y0)) * self.span * 0.5)
                * 0.5
                * (self.chord[1:] ** 2 + self.chord[:-1] ** 2)
            )
        )
        self.holizontal_tail_area = (
            self.wing_area * self.mac * self.horisontal_tail_volume / self.horisontal_tail_arm
        )
        self.vertical_tail_area = (
            self.wing_area * self.span * self.vertical_tail_volume / self.vertical_tail_arm
        )
        # aircraft
        self.drag = (
            (
                self.calibration_factor
                * (
                    self.cdS_body
                    + self.CD0 * self.wing_area
                    + self.cd_tail * (self.holizontal_tail_area + self.vertical_tail_area)
                    + self.wire_cd * self.wire_length * self.wire_diameter
                )
                + self.CDi * self.wing_area
            )
            * 0.5
            * self.rho
            * self.v_inf**2
        )
        self.CD = self.drag / (0.5 * self.rho * self.wing_area * self.v_inf**2)
        # power
        self.power = self.drag * self.v_inf / self.drivetrain_efficiency
        self.power_constraint = self.power - self.max_power

    def _stiffness(self):
        self.Ex_ply = np.zeros([self.n_div, self.max_plys])
        self.EI_ply = np.zeros([self.n_struc, self.max_plys])
        self.EA_ply = np.zeros([self.n_struc, self.max_plys])
        # siffness
        self.EI = np.zeros(self.n_struc)
        self.EA = np.zeros(self.n_struc)
        self.GIp = np.zeros(self.n_struc)

        axis = 0.0
        nui = 1 / (1 - self.nu_L * self.nu_T)
        ETL = self.E_T * self.nu_L * nui + 2 * self.G_LT
        Exx0 = self.E_L * nui
        Eyy0 = self.E_T * nui
        Exy0 = self.E_T * self.nu_L * nui
        Ex0 = Exx0 - Exy0**2 / Eyy0

        for i, plys in enumerate(self.ply_wing):
            y0_start = self.y_div[i]
            y0_end = self.y_div[i + 1]
            # plys on entire surface: plys must be symmetric
            plys_entire = plys[plys[:, 1] >= 180]
            sum_ply = np.zeros(self.n_struc)
            Exx = np.zeros(self.n_struc)
            Eyy = np.zeros(self.n_struc)
            Exy = np.zeros(self.n_struc)
            Gxy = np.zeros(self.n_struc)
            for j, ply in enumerate(plys_entire):
                axis = np.deg2rad(ply[0])
                phi = np.deg2rad(ply[1])
                mask = ply[4] * np.where(
                    (self.wing.index >= ply[2] * (y0_end - y0_start) + y0_start)
                    & (self.wing.index < ply[3] * (y0_end - y0_start) + y0_start),
                    1,
                    0,
                )
                sum_ply += mask
                Exx += mask * (
                    self.E_T * nui * np.sin(axis) ** 4
                    + 2 * ETL * (np.sin(axis) * np.cos(axis)) ** 2
                    + self.E_L * nui * np.cos(axis) ** 4
                )
                Eyy += mask * (
                    self.E_L * nui * np.sin(axis) ** 4
                    + 2 * ETL * (np.sin(axis) * np.cos(axis)) ** 2
                    + self.E_T * nui * np.cos(axis) ** 4
                )
                Exy += mask * (
                    self.E_T * self.nu_L * nui * (np.sin(axis) ** 4 + np.cos(axis) ** 4)
                    + ((self.E_L + self.E_T) * nui - 4 * self.G_LT)
                    * (np.sin(axis) * np.cos(axis)) ** 2
                )
                Gxy += mask * (
                    0.25
                    * (self.E_L + self.E_T * (1 - 2 * self.nu_L))
                    * nui
                    * np.sin(2 * axis) ** 2
                    + self.G_LT * np.cos(2 * axis) ** 2
                )
            _sum_ply = np.where(sum_ply > 0, sum_ply, np.inf)
            Exx, Eyy, Exy, Gxy = Exx / _sum_ply, Eyy / _sum_ply, Exy / _sum_ply, Gxy / _sum_ply
            Ex = Exx - Exy**2 / np.where(sum_ply > 0, Eyy, np.inf)
            R = np.where(
                (self.wing.index >= y0_start) & (self.wing.index < y0_end), 0.5 * self.diameter, 0
            )
            Iy = np.pi * sum_ply * R**3
            EA = Ex * 2 * np.pi * R * sum_ply
            self.EI += Ex * Iy
            self.EA += EA
            self.GIp += Gxy * Iy * 2

            # circumferentially partial plys (flange)
            plys_partial = plys[plys[:, 1] < 180]
            for j, ply in enumerate(plys_partial):
                axis = np.deg2rad(ply[0])
                phi = np.deg2rad(ply[1])
                if axis == 0:
                    Ex = Ex0
                else:
                    Exx = (
                        self.E_T * nui * np.sin(axis) ** 4
                        + 2 * ETL * (np.sin(axis) * np.cos(axis)) ** 2
                        + self.E_L * nui * np.cos(axis) ** 4
                    )
                    Eyy = (
                        self.E_L * nui * np.sin(axis) ** 4
                        + 2 * ETL * (np.sin(axis) * np.cos(axis)) ** 2
                        + self.E_T * nui * np.cos(axis) ** 4
                    )
                    Exy = (
                        self.E_T * self.nu_L * nui * (np.sin(axis) ** 4 + np.cos(axis) ** 4)
                        + ((self.E_L + self.E_T) * nui - 4 * self.G_LT)
                        * (np.sin(axis) * np.cos(axis)) ** 2
                    )
                    Ex = Exx - Exy**2 / Eyy
                R = np.where(
                    (self.wing.index >= ply[2] * (y0_end - y0_start) + y0_start)
                    & (self.wing.index < ply[3] * (y0_end - y0_start) + y0_start),
                    0.5 * self.diameter,
                    0,
                )
                Iy = (phi + np.sin(phi)) * ply[4] * R**3
                EA = Ex * 2 * phi * R * ply[4]
                self.EI += Ex * Iy
                self.EA += EA
                self.Ex_ply[i, j] = Ex
                self.EI_ply[:, j] += Ex * Iy
                self.EA_ply[:, j] += EA
        self.EI[-1] = self.EI[-2]
        self.EA[-1] = self.EA[-2]
        self.GIp[-1] = self.GIp[-2]

    def _compute_moment(self, y, F):
        y, F = y[::-1], F[::-1]
        return np.hstack([0.0, np.cumsum((np.cumsum(F)[:-1] + 0.5 * F[1:]) * np.abs(np.diff(y)))])[
            ::-1
        ]

    def _compute_moment_with_axial_force(self, y, F, T, EI):
        # beam-column theory
        n = len(y)
        M, S = np.zeros(n), np.zeros(n)
        L = np.abs(np.diff(y))
        for i in range(n - 2, -1, -1):
            TL_EI = T[i] * (0.5 * L[i]) ** 2 / EI[i]
            S[i] = (F[i] + (1 - TL_EI) * S[i + 1] - T[i] * L[i] * M[i + 1] / EI[i]) / (1 + TL_EI)
            M[i] = (0.5 * L[i] * F[i] + (1 - TL_EI) * M[i + 1] + L[i] * S[i + 1]) / (1 + TL_EI)
        return M, S

    def _evaluate_zerolift_deflection(self):
        self.moment_zerolift = self._compute_moment(
            y=self.y0 * self.span * 0.5, F=-self.wing_weight * self.gravity
        )
        self.M_EI_zerolift = self.moment_zerolift / self.EI
        self.deflection_angle_zerolift = np.hstack(
            [
                0.0,
                0.5
                * (self.M_EI_zerolift[1:] + self.M_EI_zerolift[:-1])
                * np.diff(self.wing.index.values)
                * self.span
                * 0.5,
            ]
        ).cumsum()
        self.deflection_zerolift = np.hstack(
            [
                0.0,
                0.5
                * (self.deflection_angle_zerolift[1:] + self.deflection_angle_zerolift[:-1])
                * np.diff(self.wing.index.values)
                * self.span
                * 0.5,
            ]
        ).cumsum()
        self.total_deflection_zerolift = self.dihedral_position + self.deflection_zerolift
        self.strain_zerolift = self.M_EI_zerolift * self.diameter * 0.5

    def _evaluate_flight_deflection(self):
        i_wire = np.argmin(np.abs(self.y0 - self.y_wire))
        z_wire_deflection = 0.0
        iter_max = self.wire_iteration if self.wire_tension > 0 else 1
        if self.wire_tension > 0:
            for i in range(iter_max):
                # wire
                wire_angle = (
                    np.arctan((self.z_wire + z_wire_deflection) / (self.y_wire * self.span * 0.5))
                    if self.y_wire > 0
                    else np.pi * 0.5
                )
                dihedral_angle_at_wire = np.arctan(
                    z_wire_deflection / (self.y_wire * self.span * 0.5)
                )
                axial_force = (
                    self.wire_tension * np.cos(wire_angle) / np.cos(dihedral_angle_at_wire)
                )
                wire_load = (
                    axial_force * np.sin(wire_angle - dihedral_angle_at_wire) / np.cos(wire_angle)
                )
                self.wire_load = np.zeros(self.n_struc)
                self.wire_load[i_wire] = wire_load
                self.axial_force = np.where(self.y0 < self.y_wire, axial_force, 0)
                # moment & deflection
                self.total_load = (
                    self.local_lift - self.wire_load - self.wing_weight * self.gravity
                )
                self.moment, _ = self._compute_moment_with_axial_force(
                    y=self.y0 * self.span * 0.5, F=self.total_load, T=self.axial_force, EI=self.EI
                )
                self.M_EI = self.moment / self.EI
                self.deflection_angle = np.hstack(
                    [
                        0.0,
                        0.5
                        * (self.M_EI[1:] + self.M_EI[:-1])
                        * np.diff(self.y0)
                        * self.span
                        * 0.5,
                    ]
                ).cumsum()
                self.deflection = np.hstack(
                    [
                        0.0,
                        0.5
                        * (self.deflection_angle[1:] + self.deflection_angle[:-1])
                        * np.diff(self.y0)
                        * self.span
                        * 0.5,
                    ]
                ).cumsum()
                self.total_deflection = self.dihedral_position + self.deflection
                z_wire_deflection = self.total_deflection[i_wire]
        else:
            # wire
            self.wire_load = np.zeros(self.n_struc)
            self.axial_force = np.zeros(self.n_struc)
            # moment & deflection
            self.total_load = self.local_lift - self.wing_weight * self.gravity
            self.moment = self._compute_moment(y=self.y0 * self.span * 0.5, F=self.total_load)
            self.M_EI = self.moment / self.EI
            self.deflection_angle = np.hstack(
                [0.0, 0.5 * (self.M_EI[1:] + self.M_EI[:-1]) * np.diff(self.y0) * self.span * 0.5]
            ).cumsum()
            self.deflection = np.hstack(
                [
                    0.0,
                    0.5
                    * (self.deflection_angle[1:] + self.deflection_angle[:-1])
                    * np.diff(self.y0)
                    * self.span
                    * 0.5,
                ]
            ).cumsum()
            self.total_deflection = self.dihedral_position + self.deflection
        self.strain = self.M_EI * self.diameter * 0.5 + self.axial_force / self.EA

    def _evaluate_twist(self):
        # simple (not practical) estimation
        # this should be computed from Cm, Cl, and change in AoA; center of pressure = 0.25 - Cm/Cl(AoA)
        delta_center_of_pressure = 0.01  # ~1%/1deg
        self.torque = (
            self.local_lift[:-1]
            * 0.5
            * (self.chord[1:] + self.chord[:-1])
            * delta_center_of_pressure
        )
        self.torque = self.torque[::-1].cumsum()[::-1]
        self.twist = np.rad2deg(
            np.hstack(
                [0.0, self.torque / (0.5 * (self.GIp[1:] + self.GIp[:-1])) * np.diff(self.y)]
            ).cumsum()
        )

    def _optimize_ply(self):
        def update_ply(j, y0_j, strain_const, y0_start, y0_end, root_limit, tip_limit):
            strain = np.abs(
                self.moment / (self.EI - self.EI_ply[:, j - 4]) * 0.5 * self.diameter
            ) + self.axial_force / (self.EA - self.EA_ply[:, j - 4])
            strain_zerolift = np.abs(
                self.moment_zerolift / (self.EI - self.EI_ply[:, j - 4]) * 0.5 * self.diameter
            )
            strain = np.max(
                [strain / self.strain_ratios[0, i], strain_zerolift / self.strain_ratios[1, i]],
                axis=0,
            )
            strain = np.where(y0_j, strain, 0)
            violated = np.where(strain >= strain_const, 1, 0)
            i_min = np.argmax(y0_j)
            i_max = len(y0_j) - np.argmax(y0_j[::-1]) - 1
            if violated.max() > 0:
                i_root = np.clip(np.argmax(violated), i_min, i_max)
                i_tip = np.clip(len(violated) - np.argmax(violated[::-1]), i_min, i_max)
            else:
                i_root, i_tip = i_min, i_min
            root = self.y0[i_root]
            tip = self.y0[i_tip]
            # move root and tip to the pipe edge if they are on the mesh points nearest to the edge
            if i_root > 0:
                if self.y0[i_root - 1] < y0_start:
                    root = y0_start
            if i_tip > 0:
                if self.y0[i_tip - 1] < y0_start:
                    tip = y0_start
            if i_tip + 1 < len(violated):
                if self.y0[i_tip + 1] >= y0_end:
                    tip = y0_end
            # inner layer must be longer than outer layer
            if tip - root > 0:
                if root > root_limit:
                    root = root_limit
                else:
                    root_limit = root
                if tip < tip_limit:
                    tip = tip_limit
                else:
                    tip_limit = tip
            return root, tip, root_limit, tip_limit

        def update(STIFFNESS=False):
            self._compute_wing_weight()
            self._compute_local_lift()
            if STIFFNESS:
                self._stiffness()
            self._evaluate_zerolift_deflection()
            self._evaluate_flight_deflection()

        # 0.99 is multiplied because strain_max can be exceeded due to weight reduction by cutting off plys
        strain_max = 0.99 * self.allowable_strain
        feasible = True
        # increase ply width to satisfy constraint
        for i in range(len(self.ply_wing) - 1, -1, -1):
            plys = self.ply_wing[i]
            n_entire = np.sum(plys[:, 1] >= 180)
            y0_i = (self.y0 >= self.y_div[i]) & (self.y0 < self.y_div[i + 1])
            strain1 = np.abs(self.strain[y0_i]).max() / self.strain_ratios[0, i]
            strain2 = np.abs(self.strain_zerolift[y0_i]).max() / self.strain_ratios[1, i]
            if max(strain1, strain2) > strain_max:
                feasible = False
                iy1 = y0_i.argmax() + np.abs(self.strain[y0_i]).argmax()
                M1 = self.moment[iy1]
                AF1 = self.axial_force[iy1]
                iy2 = y0_i.argmax() + np.abs(self.strain_zerolift[y0_i]).argmax()
                M2 = -self.moment_zerolift[iy2]
                phis = np.deg2rad(plys[n_entire:, 1])
                Ex = self.Ex_ply[i, :]
                Ex = Ex[Ex > 0]
                tr3_1 = plys[n_entire:, 4] * (0.5 * self.diameter[iy1]) ** 3
                tr_1 = plys[n_entire:, 4] * (0.5 * self.diameter[iy1])
                tr3_2 = plys[n_entire:, 4] * (0.5 * self.diameter[iy2]) ** 3
                for j in range(1, 180 - int(phis.max()), 1):
                    dphi = np.deg2rad(j)
                    dIy1 = (dphi + np.sin(phis + dphi) - np.sin(phis)) * tr3_1
                    A1 = 2 * dphi * tr_1
                    dIy2 = (dphi + np.sin(phis + dphi) - np.sin(phis)) * tr3_2
                    strain1 = (
                        np.abs(M1 / (self.EI[iy1] + np.sum(Ex * dIy1)) * 0.5 * self.diameter[iy1])
                        + AF1 / (self.EA[iy1] + np.sum(Ex * A1))
                    ) / self.strain_ratios[0, i]
                    strain2 = (
                        np.abs(M2 / (self.EI[iy2] + np.sum(Ex * dIy2)) * 0.5 * self.diameter[iy2])
                        / self.strain_ratios[1, i]
                    )
                    if max(strain1, strain2) <= strain_max:
                        break
                self.ply_wing[i][n_entire:, 1] += j
                self.ply_wing[i][n_entire:, 1] = np.clip(self.ply_wing[i][n_entire:, 1], 0, 179)
                if self.FINE_MODE:
                    update(STIFFNESS=True)
        if not feasible and not self.FINE_MODE:
            update(STIFFNESS=True)
        # cut off unnecessary ply parts in length direction
        for i in range(len(self.ply_wing) - 1, -1, -1):
            plys = self.ply_wing[i]
            y0_start = self.y_div[i]
            y0_end = self.y_div[i + 1]
            root_limit = 1.1
            tip_limit = -0.1
            flag = True
            for j in range(plys.shape[0] - 1, 3, -1):
                ply = plys[j]
                y0_j = (self.y0 >= ply[2] * (y0_end - y0_start) + y0_start) & (
                    self.y0 < ply[3] * (y0_end - y0_start) + y0_start
                )
                root, tip, root_limit, tip_limit = update_ply(
                    j, y0_j, strain_max, y0_start, y0_end, root_limit, tip_limit
                )
                # removed weight should be considered in strain evaluarion if many layers are removed at once
                if flag and tip > root:
                    flag = False
                    if plys.shape[0] - 1 - j > 4:
                        update(STIFFNESS=False)
                        root, tip, root_limit, tip_limit = update_ply(
                            j, y0_j, strain_max, y0_start, y0_end, root_limit, tip_limit
                        )
                # update ply and stiffness
                self.ply_wing[i][j, 2] = (root - y0_start) / (y0_end - y0_start)
                self.ply_wing[i][j, 3] = (tip - y0_start) / (y0_end - y0_start)
                y0_off = y0_j & ((self.y0 < root) | (self.y0 >= tip))
                self.EI[y0_off] -= self.EI_ply[y0_off, j - 4]
                self.EA[y0_off] -= self.EA_ply[y0_off, j - 4]
                self.EI[-1] = self.EI[-2]
                self.EA[-1] = self.EA[-2]
                if self.FINE_MODE:
                    update(STIFFNESS=False)
            # root or tip must be at the pip edge: this part can be ignored for practical design
            not0 = self.ply_wing[i][:, 3] > self.ply_wing[i][:, 2]
            sum_root = self.ply_wing[i][not0, 2].sum()
            sum_tip = (1 - self.ply_wing[i][not0, 2]).sum()
            if sum_root * sum_tip > 0:
                y0_j = (self.y0 >= y0_start) & (self.y0 < y0_end)
                if sum_root <= sum_tip:
                    for j in range(self.ply_wing[i].shape[0] - 1, 3, -1):
                        if (self.ply_wing[i][j, 3] > self.ply_wing[i][j, 2]) and (
                            self.ply_wing[i][j, 2] > 0
                        ):
                            root = self.ply_wing[i][j, 2] * (y0_end - y0_start) + y0_start
                            y0_on = y0_j & (self.y0 < root)
                            self.EI[y0_on] += self.EI_ply[y0_on, j - 4]
                            self.EA[y0_on] += self.EA_ply[y0_on, j - 4]
                            self.EI[-1] = self.EI[-2]
                            self.EA[-1] = self.EA[-2]
                    self.ply_wing[i][not0, 2] = 0.0
                else:
                    for j in range(self.ply_wing[i].shape[0] - 1, 3, -1):
                        if (self.ply_wing[i][j, 3] > self.ply_wing[i][j, 2]) and (
                            self.ply_wing[i][j, 3] < 1
                        ):
                            tip = self.ply_wing[i][j, 3] * (y0_end - y0_start) + y0_start
                            y0_on = y0_j & (self.y0 >= tip)
                            self.EI[y0_on] += self.EI_ply[y0_on, j - 4]
                            self.EA[y0_on] += self.EA_ply[y0_on, j - 4]
                            self.EI[-1] = self.EI[-2]
                            self.EA[-1] = self.EA[-2]
                    self.ply_wing[i][not0, 3] = 1.0
            if not self.FINE_MODE:
                update(STIFFNESS=False)

    def _compute_constraints(self):
        self.max_strain = max(np.abs(self.strain).max(), np.abs(self.strain_zerolift).max())
        self.wing_tip_deflection = self.total_deflection[-1]
        self.zerolift_deflection = self.total_deflection_zerolift[-1]
        self.strain_constraint = self.max_strain / self.allowable_strain - 1
        self.allowable_deflection = (
            0.5 * self.span * np.sin(np.deg2rad(self.max_dihedral_angle_at_tip))
        )
        self.deflection_constraint = self.wing_tip_deflection - self.allowable_deflection
        self.max_twist = np.abs(self.twist).max()
        self.speed_constraint = 1 - (self.v_inf / self.v_min) ** 3

    def _update_wing_table(self):
        # for outputing design data
        if self.FINE_MODE:
            self.wing["re"] = self.re
            self.wing["cd0"] = self.cd0
        self.wing["weight"] = self.wing_weight
        self.wing["local lift"] = self.local_lift
        self.wing["dihedral position"] = self.dihedral_position
        self.wing["M"] = self.moment
        self.wing["EI"] = self.EI
        self.wing["EA"] = self.EA
        self.wing["M_zerolift"] = self.moment_zerolift
        self.wing["M_EI_zerolift"] = self.M_EI_zerolift
        self.wing["zerolift deflection angle"] = self.deflection_angle_zerolift
        self.wing["zerolift deflection"] = self.deflection_zerolift
        self.wing["total zerolift deflection"] = self.total_deflection_zerolift
        self.wing["zerolift strain"] = self.strain_zerolift
        self.wing["strain"] = self.strain
        self.wing["wire load"] = self.wire_load
        self.wing["axial force"] = self.axial_force
        self.wing["total load"] = self.total_load
        self.wing["M_EI"] = self.M_EI
        self.wing["deflection angle"] = self.deflection_angle
        self.wing["deflection"] = self.deflection
        self.wing["total deflection"] = self.total_deflection
