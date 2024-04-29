# Author: Jongwon Lee
# This file contains code for two-view reconstruction with inertial data

import sys
import cv2
import sfm
import numpy as np
import matplotlib.pyplot as plt
import rerun as rr
from scipy.spatial.transform import Rotation

import symforce
# symforce.set_epsilon_to_symbol()
import symforce.symbolic as sf
from symforce.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from symforce.opt.noise_models import PseudoHuberNoiseModel
from symforce.opt.noise_models import BarronNoiseModel
import sym

import sfm
import imu_preintegration


"""
Helper functions for math with Symforce compatibility
"""
def sf_Hat(w: sf.V3) -> sf.Matrix33:
    """
    Construct a 3x3 skew symmetric matrix from an array with length three (i.e., "hat" operation)
    https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
    """
    x, y, z = w

    Hat = sf.Matrix33([[0, -z, y],
                       [z, 0, -x],
                       [-y, x, 0]])

    return Hat


def sf_Vee(R: sf.Matrix33) -> sf.V3:
    """
    Construct an array with length three from a 3x3 skew symmetric matrix (i.e., "vee" operation)
    https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
    """
    Vee = sf.V3([R[2,1], -R[2,0], R[1,0]])

    return Vee


def sf_logm(M: sf.Matrix33, ord=5) -> sf.Matrix33:
    """
    Compute the matrix logarithm using the Taylor series expansion.
    https://en.wikipedia.org/wiki/Logarithm_of_a_matrix

    Parameters:
    - M (sf.Matrix33): A 3x3 matrix, close to the identity matrix for convergence.
    - ord (int): The number of terms in the Taylor series to compute.

    Returns:
    - sf.Matrix33: The matrix logarithm of M.

    Note:
    - The series converges when M is close enough to the identity matrix.
    - Consider dynamically adjusting 'ord' based on the norm of (M - I) for better accuracy.
    """
    I = sf.Matrix.eye(3)  # Identity matrix
    delta_M = M - I       # Compute M - I once and reuse it
    pow = delta_M         # Start with the first power of M - I
    log_M = sf.Matrix.zeros(3, 3)  # Initialize the result to zero

    for k in range(1, ord + 1):
        term = pow / k * (-1) ** (k + 1)  # Calculate the current term
        log_M += term                    # Add the current term to the sum
        pow = pow * delta_M              # Update pow to the next power of M - I

        # Optional: Add a convergence check here to break early if terms are very small

    return log_M


"""
Residuals associated with inetial data
"""
# See Eq. 37 in Forster or Eq.A.21 in Supplementary
def imu_preintegration_residual(
        # Keys subject to optimize
        pose_i: sf.Pose3,
        vel_i: sf.V3,
        pose_j: sf.Pose3,
        vel_j: sf.V3,
        # acc_bias_i: sf.V3,
        # gyr_bias_i: sf.V3,
        # IMU preintegration results
        dR_ij: sf.Matrix33,
        dv_ij: sf.V3,
        dp_ij: sf.V3,
        sqrt_info_ij: sf.Matrix99,
        dt_ij: sf.Scalar,
        gravity: sf.V3,
        # Epsilon
        epsilon: sf.Scalar,
        ) -> sf.V9:
    R_i = pose_i.R.to_rotation_matrix()
    p_i = pose_i.t

    R_j = pose_j.R.to_rotation_matrix()
    p_j = pose_j.t

    # Refer to Eq. 37 in Forster or Eq.A.21 in Supplementary
    residual_dR_ij = sf_Vee(sf_logm(dR_ij.T * R_i.T * R_j))
    residual_dv_ij = R_i.T * (vel_j - vel_i - gravity * dt_ij) - dv_ij
    residual_dp_ij = R_i.T * (p_j - p_i - vel_i * dt_ij - 0.5 * gravity * dt_ij ** 2) - dp_ij

    return sqrt_info_ij * sf.V9(residual_dR_ij.col_join(residual_dv_ij).col_join(residual_dp_ij))


# See Eq. 40 in Forster
def imu_bias_residual(
        # Keys subject to optimize
        acc_bias_i: sf.V3,
        gyr_bias_i: sf.V3,
        acc_bias_j: sf.V3,
        gyr_bias_j: sf.V3,
        acc_cov: sf.Scalar,
        gyr_cov: sf.Scalar,
        epsilon: sf.Scalar,
        ) -> sf.V6:
    # Compute the square root of the inverse covariance directly
    gyr_info = 1 / sf.sqrt(gyr_cov)
    acc_info = 1 / sf.sqrt(acc_cov)

    # Create a diagonal matrix with these values
    sqrt_info = sf.Matrix.diag(sf.Matrix([gyr_info, gyr_info, gyr_info, acc_info, acc_info, acc_info]))

    return sqrt_info * sf.V6((gyr_bias_j - gyr_bias_i).col_join(acc_bias_j - acc_bias_i))


"""
Functions for optimization
"""
def get_optimizer(views, tracks, acc_meas, gyr_meas, K):
    """
    Returns a symforce optimizer and a set of initial_values that
    allow you to solve the bundle adjustment problem corresponding
    to the given views and tracks.
    NOTE: This function only allows optimization for two-view reconstruction with inertial data.
    Further improvement (multi-view reconstruction with inertial data) is out of scope at the moment.
    """

    # Create data structures
    initial_values = Values(
        fx=K[0,0],
        fy=K[1,1],
        cx=K[0,2],
        cy=K[1,2],
        tracks=[],
        epsilon=sym.epsilon,
    )
    optimized_keys = []
    factors = []

    # For each view that has a pose estimate, add this pose estimate as an initial
    # value and (if not the first view) as an optimized key.
    #print(f'Iterate over {len(views)} views:')
    for i, view in enumerate(views):
        if (view['R_inB_ofA'] is None) or (view['p_inB_ofA'] is None):
            continue
        
        initial_values[f'T_inB{i}_ofA'] = sym.Pose3(
            R=sym.Rot3.from_rotation_matrix(view['R_inB_ofA']),
            t=view['p_inB_ofA'],
        )

        if i > 0:
            optimized_keys.append(f'T_inB{i}_ofA')
            #print(f' T_inB{i}_ofA has an initial value and is an optimized key')
        #else:
        #    print(f' T_inB{i}_ofA has an initial value')

    # Add a factor to fix the scale (the relative distance between frames B0 and
    # B1 will be something close to one).
    ### NOTE: Below is commented out for IMU setup
    #print(f'T_inB{1}_ofA has an sf_scale_residual factor')
    # factors = [
    #     Factor(
    #         residual=sfm.sf_scale_residual,
    #         keys=[
    #             f'T_inB{1}_ofA',
    #             'epsilon',
    #         ],
    #     )
    # ]

    # For each valid track, add its 3d point as an initial value and an optimized
    # key, and then, for each match in this track, add its 2d point as an initial
    # value and add a factor to penalize reprojection error.
    #print(f'Iterate over {len(tracks)} tracks:')
    for i_track, track in enumerate(tracks):
        if not track['valid']:
            continue
        
        #if (i_track == 0) or (i_track == len(tracks) - 1):
        #    print(f' track {i_track}:')
        #    print(f'  track_{i_track}_p_inA has an initial value and is an optimized key')
        #elif (i_track == 1):
        #    print('\n ...\n')
        initial_values[f'track_{i_track}_p_inA'] = track['p_inA']
        optimized_keys.append(f'track_{i_track}_p_inA')

        for match in track['matches']:
            view_id = match['view_id']
            feature_id = match['feature_id']
            #if (i_track == 0) or (i_track == len(tracks) - 1):
            #    print(f'  track_{i_track}_b_{view_id} has an initial value and an sf_projection_residual factor')
            initial_values[f'track_{i_track}_b_{view_id}'] = views[view_id]['pts'][feature_id]['pt2d']
            factors.append(Factor(
                residual=sfm.sf_projection_residual,
                keys=[
                    f'T_inB{view_id}_ofA',
                    f'track_{i_track}_p_inA',
                    f'track_{i_track}_b_{view_id}',
                    'fx',
                    'fy',
                    'cx',
                    'cy',
                    'epsilon',
                ],
            ))
    
    ### NOTE: Incorporating IMU starts here

    # Step 0: Define required parameters for IMUs
    sigma_acc_wn = 1e-4  # accelerometer white noise sigma
    sigma_gyr_wn = 1e-6  # gyroscope white noise sigma
    sigma_acc_rw = 1e-5  # accelerometer random walk sigma
    sigma_gyr_rw = 1e-7  # gyroscope random walk sigma

    dt = 0.1

    Cov_acc_bias = sigma_acc_rw ** 2  * dt
    Cov_gyr_bias = sigma_gyr_rw ** 2  * dt

    # Step 1: Perform IMU preintegration
    integrator = imu_preintegration.Preintegrator(dt, sigma_acc_wn**2, sigma_gyr_wn**2)        
    # integrator.set_bias(acc_bias, gyr_bias)
    dR, dv, dp, Sigma = integrator.integrate(acc_meas, gyr_meas)
    chol = np.linalg.cholesky(np.linalg.inv(Sigma))
    sqrt_info = chol.T
    
    # Step 2: Add required initial values
    # Below are variables being optimized
    for i, _ in enumerate(views):
        initial_values[f'vel_{i}'] = np.zeros(3)
        initial_values[f'acc_bias_{i}'] = np.zeros(3)
        initial_values[f'gyr_bias_{i}'] = np.zeros(3)

        optimized_keys.append(f'vel_{i}')
        optimized_keys.append(f'acc_bias_{i}')
        optimized_keys.append(f'gyr_bias_{i}')

    # Below are values remaining constant
    assert(len(views) == 2)
    for i in range(len(views)-1):
        initial_values[f'dR_{i}{i+1}'] = dR
        initial_values[f'dv_{i}{i+1}'] = dv
        initial_values[f'dp_{i}{i+1}'] = dp
        initial_values[f'sqrt_info_{i}{i+1}'] = sqrt_info
        initial_values[f'dt_{i}{i+1}'] = dt * len(acc_meas)
    
    initial_values['gravity'] = np.array([0., 0., -9.81])
    initial_values['Cov_acc_bias'] = Cov_acc_bias
    initial_values['Cov_gyr_bias'] = Cov_gyr_bias

    # Step 3-1: Add a residual associating poses and velocities between frames
    factors.append(Factor(
                residual=imu_preintegration_residual,
                keys=[
                    'T_inB0_ofA', 'vel_0',
                    'T_inB1_ofA', 'vel_1',
                    'dR_01',
                    'dv_01',
                    'dp_01',
                    'sqrt_info_01',
                    'dt_01',
                    'gravity',
                    'epsilon',
                ],
            ))
    
    # Step 3-2: Add a residual associating biases between frames
    factors.append(Factor(
                residual=imu_bias_residual,
                keys=[
                    'acc_bias_0',
                    'gyr_bias_0',
                    'acc_bias_1',
                    'gyr_bias_1',
                    'Cov_acc_bias',
                    'Cov_gyr_bias',
                    'epsilon',
                ],
            ))

    ### NOTE: Incorporating IMU ends here

    # Create optimizer
    optimizer = Optimizer(
        factors=factors,
        optimized_keys=optimized_keys,
        debug_stats=True,
        params=Optimizer.Params(
            iterations=100,
            use_diagonal_damping=True,
            lambda_down_factor=0.1,
            lambda_up_factor=5.,
            early_exit_min_reduction=1e-4,
        ),
    )

    return optimizer, initial_values


def vio_nonlinear_optimize(views, tracks, acc_meas, gyr_meas, K, max_reprojection_err):
    """ Perform non-linear optimization on views and tracks
    
    :param views: views to optimize over
    :param tracks: tracks to optimizer
    :param acc_meas: accelerometer measurements
    :param gyr_meas: gyroscope measurements
    :param K: camera matrix (shape: (3,3))
    :param max_reprojection_err: maximum reprojection error
    :returns: views, tracks
    """
    optimizer, initial_values = get_optimizer(views, tracks, acc_meas, gyr_meas, K)
    result = optimizer.optimize(initial_values)
    new_views, new_tracks = sfm.copy_results(views, tracks)
    assert(result.status == symforce.opt.optimizer.Optimizer.Status.SUCCESS)
    # Modifies views and tracks in-place
    sfm.store_results(new_views, new_tracks, K, result, max_reprojection_err=max_reprojection_err)
    return new_views, new_tracks