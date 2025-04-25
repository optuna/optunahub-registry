import numpy as np

from .designer import HPADesigner


### Unconstrained problems ########################################################
class HPA101(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=10,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=False,
            PAYLOAD=False,
        )
        self.nf = 1
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        g = 10 * penalty1 + 2 * penalty2 + 2 * penalty3
        f = np.array([self.drag + g])
        return f


class HPA102(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=False,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 1
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        penalty4 = max(0.0, self.speed_constraint)
        g = 10 * penalty1 + 2 * penalty2 + 2 * penalty3 + 10 * penalty4
        f = np.array([self.power + 10 * g])
        return f


class HPA103(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=False,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 1
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        penalty4 = max(0.0, self.power_constraint)
        g = 10 * penalty1 + 2 * penalty2 + 2 * penalty3 + 0.1 * penalty4
        f = np.array([-self.v_inf + g])
        return f


class HPA201(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=10,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 2
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        penalty4 = max(0.0, self.power_constraint)
        g = 10 * penalty1 + 2 * penalty2 + 2 * penalty3 + 0.1 * penalty4
        f = np.array([self.power + 10 * g, -self.v_inf + g])
        return f


class HPA202(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=False,
            DIHEDRAL=False,
            PAYLOAD=False,
        )
        self.nf = 2
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED
        self.max_dihedral_angle_at_root = 0.0  # [deg]

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, self.speed_constraint)
        g = 10 * penalty1 + 10 * penalty2
        f = np.array(
            [
                self.power + 10 * g,
                max(np.abs(self.wing_tip_deflection), np.abs(self.zerolift_deflection)) + 0.5 * g,
            ]
        )
        return f


class HPA203(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=True,
        )
        self.nf = 2
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        g = 10 * penalty1 + 2 * penalty2 + 2 * penalty3
        f = np.array([self.drag + g, -self.payload + 10 * g])
        return f


class HPA204(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 2
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        penalty4 = max(0.0, self.power_constraint)
        g = 10 * penalty1 + 2 * penalty2 + 2 * penalty3 + 0.1 * penalty4
        f = np.array([-self.v_inf + g, self.max_twist + 0.1 * g])
        return f


class HPA205(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 2
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        penalty4 = max(0.0, self.power_constraint)
        g = 10 * penalty1 + 2 * penalty2 + 2 * penalty3 + 0.1 * penalty4
        f = np.array([-self.v_inf + g, self.wire_tension + 200 * g])
        return f


class HPA301(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=10,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 3
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        penalty4 = max(0.0, self.power_constraint)
        g = 10 * penalty1 + 2 * penalty2 + 2 * penalty3 + 0.1 * penalty4
        f = np.array([self.power + 10 * g, -self.v_inf + g, -self.wing_efficiency + 0.1 * g])
        return f


class HPA302(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=False,
            DIHEDRAL=False,
            PAYLOAD=False,
        )
        self.nf = 3
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED
        self.max_dihedral_angle_at_root = 0.0  # [deg]

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, self.speed_constraint)
        g = 10 * penalty1 + 10 * penalty2
        f = np.array(
            [
                self.power + 10 * g,
                max(np.abs(self.wing_tip_deflection), np.abs(self.zerolift_deflection)) + 0.5 * g,
                self.max_twist + 0.1 * g,
            ]
        )
        return f


class HPA303(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=True,
        )
        self.nf = 3
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        g = 10 * penalty1 + 2 * penalty2 + 2 * penalty3
        f = np.array([self.drag + g, self.y_aoa[0] + g, -self.payload + 10 * g])
        return f


class HPA304(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 3
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        penalty4 = max(0.0, self.power_constraint)
        g = 10 * penalty1 + 2 * penalty2 + 2 * penalty3 + 0.1 * penalty4
        f = np.array([self.drag + g, -self.v_inf + g, self.max_twist + 0.1 * g])
        return f


class HPA305(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 3
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        penalty4 = max(0.0, self.power_constraint)
        g = 10 * penalty1 + 2 * penalty2 + 2 * penalty3 + 0.1 * penalty4
        f = np.array([self.power + 10 * g, -self.v_inf + g, self.wire_tension + 200 * g])
        return f


class HPA401(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 4
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        penalty4 = max(0.0, self.power_constraint)
        g = 10 * penalty1 + 2 * penalty2 + 2 * penalty3 + 0.1 * penalty4
        f = np.array(
            [self.drag + g, -self.v_inf + g, self.max_twist + 0.1 * g, self.empty_weight + g]
        )
        return f


class HPA402(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=False,
            DIHEDRAL=False,
            PAYLOAD=False,
        )
        self.nf = 4
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED
        self.max_dihedral_angle_at_root = 0.0  # [deg]

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, self.speed_constraint)
        g = 10 * penalty1 + 10 * penalty2
        f = np.array(
            [
                self.power + 10 * g,
                max(np.abs(self.wing_tip_deflection), np.abs(self.zerolift_deflection)) + 0.5 * g,
                self.max_twist + 0.1 * g,
                self.span + g,
            ]
        )
        return f


class HPA403(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 4
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        penalty4 = max(0.0, self.power_constraint)
        g = 10 * penalty1 + 2 * penalty2 + 2 * penalty3 + 0.1 * penalty4
        f = np.array(
            [
                -self.v_inf + g,
                -self.wing_efficiency + 0.1 * g,
                self.empty_weight + g,
                self.wire_tension + 200 * g,
            ]
        )
        return f


class HPA501(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 5
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        penalty4 = max(0.0, self.power_constraint)
        g = 10 * penalty1 + 2 * penalty2 + 2 * penalty3 + 0.1 * penalty4
        f = np.array(
            [
                self.drag + g,
                -self.v_inf + g,
                self.max_twist + 0.1 * g,
                -self.wing_efficiency + 0.1 * g,
                self.empty_weight + g,
            ]
        )
        return f


class HPA502(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=True,
        )
        self.nf = 5
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        penalty4 = max(0.0, self.power_constraint)
        g = 10 * penalty1 + 2 * penalty2 + 2 * penalty3 + 0.1 * penalty4
        f = np.array(
            [
                self.power + 10 * g,
                -self.v_inf + g,
                self.y_aoa[0] + g,
                self.wire_tension + 200 * g,
                -self.payload + 10 * g,
            ]
        )
        return f


class HPA601(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 6
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        penalty4 = max(0.0, self.power_constraint)
        g = 10 * penalty1 + 2 * penalty2 + 2 * penalty3 + 0.1 * penalty4
        f = np.array(
            [
                self.power + 10 * g,
                -self.v_inf + g,
                self.max_twist + 0.1 * g,
                -self.wing_efficiency + 0.1 * g,
                self.empty_weight + g,
                self.span + g,
            ]
        )
        return f


class HPA901(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=True,
        )
        self.nf = 9
        self.ng = 0
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        penalty1 = max(0.0, self.strain_constraint)
        penalty2 = max(0.0, -self.zerolift_deflection)
        penalty3 = max(0.0, self.deflection_constraint)
        penalty4 = max(0.0, self.power_constraint)
        g = 10 * penalty1 + 2 * penalty2 + 2 * penalty3 + 0.1 * penalty4
        f = np.array(
            [
                self.power + 10 * g,
                -self.v_inf + g,
                self.max_twist + 0.1 * g,
                -self.wing_efficiency + 0.1 * g,
                self.empty_weight + g,
                self.span + g,
                self.y_aoa[0] + g,
                self.wire_tension + 200 * g,
                -self.payload + 10 * g,
            ]
        )
        return f


### Constrained problems ########################################################
class HPA131(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=10,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=False,
            PAYLOAD=False,
        )
        self.nf = 1
        self.ng = 3
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array([self.drag])
        g = np.array(
            [self.strain_constraint, -self.zerolift_deflection, self.deflection_constraint]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA142(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=False,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 1
        self.ng = 4
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array([self.power])
        g = np.array(
            [
                self.strain_constraint,
                -self.zerolift_deflection,
                self.deflection_constraint,
                self.speed_constraint,
            ]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA143(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=False,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 1
        self.ng = 4
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array([-self.v_inf])
        g = np.array(
            [
                self.strain_constraint,
                -self.zerolift_deflection,
                self.deflection_constraint,
                self.power_constraint,
            ]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA241(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=10,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 2
        self.ng = 4
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array([self.power, -self.v_inf])
        g = np.array(
            [
                self.strain_constraint,
                -self.zerolift_deflection,
                self.deflection_constraint,
                self.power_constraint,
            ]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA222(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=False,
            DIHEDRAL=False,
            PAYLOAD=False,
        )
        self.nf = 2
        self.ng = 2
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED
        self.max_dihedral_angle_at_root = 0.0  # [deg]

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array(
            [self.power, max(np.abs(self.wing_tip_deflection), np.abs(self.zerolift_deflection))]
        )
        g = np.array(
            [self.strain_constraint, self.speed_constraint]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA233(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=True,
        )
        self.nf = 2
        self.ng = 3
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array([self.drag, -self.payload])
        g = np.array(
            [self.strain_constraint, -self.zerolift_deflection, self.deflection_constraint]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA244(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 2
        self.ng = 4
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array([-self.v_inf, self.max_twist])
        g = np.array(
            [
                self.strain_constraint,
                -self.zerolift_deflection,
                self.deflection_constraint,
                self.power_constraint,
            ]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA245(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 2
        self.ng = 4
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array([-self.v_inf, self.wire_tension])
        g = np.array(
            [
                self.strain_constraint,
                -self.zerolift_deflection,
                self.deflection_constraint,
                self.power_constraint,
            ]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA341(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=10,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 3
        self.ng = 4
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array([self.power, -self.v_inf, -self.wing_efficiency])
        g = np.array(
            [
                self.strain_constraint,
                -self.zerolift_deflection,
                self.deflection_constraint,
                self.power_constraint,
            ]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA322(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=False,
            DIHEDRAL=False,
            PAYLOAD=False,
        )
        self.nf = 3
        self.ng = 2
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED
        self.max_dihedral_angle_at_root = 0.0  # [deg]

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array(
            [
                self.power,
                max(np.abs(self.wing_tip_deflection), np.abs(self.zerolift_deflection)),
                self.max_twist,
            ]
        )
        g = np.array(
            [self.strain_constraint, self.speed_constraint]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA333(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=True,
        )
        self.nf = 3
        self.ng = 3
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array([self.drag, self.y_aoa[0], -self.payload])
        g = np.array(
            [self.strain_constraint, -self.zerolift_deflection, self.deflection_constraint]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA344(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 3
        self.ng = 4
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array([self.drag, -self.v_inf, self.max_twist])
        g = np.array(
            [
                self.strain_constraint,
                -self.zerolift_deflection,
                self.deflection_constraint,
                self.power_constraint,
            ]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA345(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 3
        self.ng = 4
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array([self.power, -self.v_inf, self.wire_tension])
        g = np.array(
            [
                self.strain_constraint,
                -self.zerolift_deflection,
                self.deflection_constraint,
                self.power_constraint,
            ]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA441(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 4
        self.ng = 4
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array([self.drag, -self.v_inf, self.max_twist, self.empty_weight])
        g = np.array(
            [
                self.strain_constraint,
                -self.zerolift_deflection,
                self.deflection_constraint,
                self.power_constraint,
            ]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA422(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=False,
            DIHEDRAL=False,
            PAYLOAD=False,
        )
        self.nf = 4
        self.ng = 2
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED
        self.max_dihedral_angle_at_root = 0.0  # [deg]

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array(
            [
                self.power,
                max(np.abs(self.wing_tip_deflection), np.abs(self.zerolift_deflection)),
                self.max_twist,
                self.span,
            ]
        )
        g = np.array(
            [self.strain_constraint, self.speed_constraint]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA443(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 4
        self.ng = 4
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array([-self.v_inf, -self.wing_efficiency, self.empty_weight, self.wire_tension])
        g = np.array(
            [
                self.strain_constraint,
                -self.zerolift_deflection,
                self.deflection_constraint,
                self.power_constraint,
            ]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA541(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 5
        self.ng = 4
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array(
            [self.drag, -self.v_inf, self.max_twist, -self.wing_efficiency, self.empty_weight]
        )
        g = np.array(
            [
                self.strain_constraint,
                -self.zerolift_deflection,
                self.deflection_constraint,
                self.power_constraint,
            ]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA542(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=True,
        )
        self.nf = 5
        self.ng = 4
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array([self.power, -self.v_inf, self.y_aoa[0], self.wire_tension, -self.payload])
        g = np.array(
            [
                self.strain_constraint,
                -self.zerolift_deflection,
                self.deflection_constraint,
                self.power_constraint,
            ]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA641(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=False,
        )
        self.nf = 6
        self.ng = 4
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array(
            [
                self.power,
                -self.v_inf,
                self.max_twist,
                -self.wing_efficiency,
                self.empty_weight,
                self.span,
            ]
        )
        g = np.array(
            [
                self.strain_constraint,
                -self.zerolift_deflection,
                self.deflection_constraint,
                self.power_constraint,
            ]
        )  # g<=0 means constraints are satisfied
        return f, g


class HPA941(HPADesigner):
    def __init__(self, n_div=4, level=0, NORMALIZED=True):
        assert n_div > 0 and isinstance(n_div, int)
        assert 0 <= level <= 2 and isinstance(level, int)
        AIRFOIL = False if level == 0 else True
        super().__init__(
            n_div=n_div,
            max_plys=20,
            level=level,
            AIRFOIL=AIRFOIL,
            WIRE=True,
            DIHEDRAL=True,
            PAYLOAD=True,
        )
        self.nf = 9
        self.ng = 4
        self.nx = self.n_x
        self.NORMALIZED = NORMALIZED

    def __call__(self, x):
        x = np.array(x)
        self.evaluate_performance_from_x(x, self.NORMALIZED)
        f = np.array(
            [
                self.power,
                -self.v_inf,
                self.max_twist,
                -self.wing_efficiency,
                self.empty_weight,
                self.span,
                self.y_aoa[0],
                self.wire_tension,
                -self.payload,
            ]
        )
        g = np.array(
            [
                self.strain_constraint,
                -self.zerolift_deflection,
                self.deflection_constraint,
                self.power_constraint,
            ]
        )  # g<=0 means constraints are satisfied
        return f, g
