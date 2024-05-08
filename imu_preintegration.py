# Author: Jongwon Lee
# This file contains code for IMU preintegration

# For numerical methods
import numpy as np
from scipy.linalg import expm, logm
from scipy.spatial.transform import Rotation as R

# For image processing and visualization of results
import matplotlib.pyplot as plt

# For optimization with symforce
import symforce
# symforce.set_epsilon_to_symbol()
import symforce.symbolic as sf
from symforce.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
import sym

"""
Helper functions
"""
def Hat(w):
    """
    Construct a 3x3 skew symmetric matrix from an array with length three (i.e., "hat" operation)
    https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
    """
    assert(len(w) == 3)

    x, y, z = w

    Hat = np.array([[0, -z, y],
                    [z, 0, -x],
                    [-y, x, 0]])

    return Hat

def Vee(R):
    """
    Construct an array with length three from a 3x3 skew symmetric matrix (i.e., "vee" operation)
    https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
    """
    assert(R.shape == (3,3))

    Vee = np.array([R[2,1], -R[2,0], R[1,0]])

    return Vee

def J_SO3(phi):
    """
    Refer to Eq. 8 in Forster
    """
    assert(len(phi) == 3)

    phi_norm = np.linalg.norm(phi)

    return np.eye(3) \
    - (1 - np.cos(phi_norm)) / phi_norm ** 2 * Hat(phi) \
    + (phi_norm - np.sin(phi_norm)) / phi_norm ** 3 * Hat(phi) * Hat(phi)

"""
IMU preintegrator class
"""
class Preintegrator:
    def __init__(self,
                 dt,
                 acc_cov, gyr_cov,
                 acc_bias=np.zeros(3), gyr_bias=np.zeros(3),
                 ):
        """ Creates pre-integrator
        
        :param dt: delta time
        :param acc_cov: covariance for accelerometer
        :param gyr_cov: covariance for gyro
        :param acc_bias: accelerometer bias
        :param gyr_bias: gyroscope bias
        """
        self.dt = dt

        self.Sigma_eta = np.block([[gyr_cov * np.eye(3),        np.zeros((3,3))],
                                   [    np.zeros((3,3)),    acc_cov * np.eye(3)]]) / self.dt
        self.Sigma = None

        self.acc_bias = acc_bias
        self.gyr_bias = gyr_bias

    def set_bias(self, acc_bias, gyr_bias):
        """ Sets the bias for acccelerometer and gyroscope """
        self.acc_bias = acc_bias
        self.gyr_bias = gyr_bias

    def set_cov(self, Cov):
        """ Sets the covariance """
        self.Sigma = Cov

    def integrate(self, acc_meas, gyr_meas):
        """ Perform integration

        :param acc_meas: array of accelerometer measurements
        :param gyr_meas: array of gyroscope measurements
        
        :returns: dR, dv, dp, and Sigma
            - dR is change in rotation
            - dv is change in velocity
            - dp is change in position
        """
        assert(len(acc_meas) == len(gyr_meas))

        Sigma = np.zeros((9,9)) if self.Sigma is None else self.Sigma
        dR = np.eye(3)
        dv = np.zeros(3)
        dp = np.zeros(3)

        for acc, gyr in zip(acc_meas, gyr_meas):
            Sigma_i_k = Sigma.copy()
            dR_i_k = dR.copy()
            dv_i_k = dv.copy()
            dp_i_k = dp.copy()

            dR_k_k1 = expm(Hat((gyr - self.gyr_bias) * self.dt))

            # Propagate covariance
            # Refer to Eq.A.9 in Forster supplementary
            A = np.block([[dR_k_k1.T, np.zeros((3,3)), np.zeros((3,3))],
                          [- dR_i_k @ Hat(acc - self.acc_bias) * self.dt, np.eye(3), np.zeros((3,3))],
                          [- 0.5 * dR_i_k @ Hat(acc - self.acc_bias) * self.dt ** 2, np.eye(3) * self.dt, np.eye(3)],
                          ])
            B = np.block([[J_SO3((gyr - self.gyr_bias) * self.dt), np.zeros((3,3))],
                          [np.zeros((3,3)), dR_i_k * self.dt],
                          [np.zeros((3,3)), 0.5 * dR_i_k * self.dt ** 2]])

            Sigma = A @ Sigma_i_k @ A.T + B @ self.Sigma_eta @ B.T

            # Propagate increment of pose and velocity
            # Refer to Eq.A.10 in Forster supplementary
            dR = dR_i_k @ dR_k_k1
            dv = dv_i_k + dR_i_k @ (acc - self.acc_bias) * self.dt
            dp = dp_i_k + dv_i_k * self.dt + 0.5 * dR_i_k @ (acc - self.acc_bias) * self.dt ** 2

        return dR, dv, dp, Sigma


if __name__ == "__main__":
    sigma_acc_wn = 1e-4  # accelerometer white noise sigma
    sigma_gyr_wn = 1e-6  # gyroscope white noise sigma
    sigma_acc_rw = 1e-5  # accelerometer random walk sigma
    sigma_gyr_rw = 1e-7  # gyroscope random walk sigma

    dt = 0.1

    Cov_acc_bias = sigma_acc_rw ** 2  * dt
    Cov_gyr_bias = sigma_gyr_rw ** 2  * dt

    integrator = Preintegrator(dt, sigma_acc_wn**2, sigma_gyr_wn**2)

    # Generate simulated imu data
    acc_meas = np.array([0., 0., 9.81]*10).reshape(10,3) + np.random.random((10,3)) / 1e3
    gyr_meas = np.random.random((10,3)) / 1e3

    # Perform IMU preintegration
    dR, dv, dp, Sigma = integrator.integrate(acc_meas, gyr_meas)