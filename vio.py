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


def undistort(pts2d, K, distortion, n_iter=5):
    """ Undistort the provided points
    
    :param pts2d: points to undistort (shape: (N,2))
    :param K: camera matrix (shape: (3,3))
    :param distortion: distortion parameters (list of length 4)
    :param n_iter: number of iterations for refinement
    :return: undistorted points (shape: (N,1,2))
    """
    # get intrinsics
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    # get distortion parameters
    k1, k2, p1, p2 = distortion

    # undistort
    # used https://yangyushi.github.io/code/2020/03/04/opencv-undistort.html as reference.
    xs = (pts2d[:,0] - cx) / fx
    ys = (pts2d[:,1] - cy) / fy
    xs0 = xs.copy()
    ys0 = ys.copy()
    # iteratively refine
    for iteration_num in range(n_iter):
        rad_sqr = np.square(xs) + np.square(ys)
        k_factor =  1 / (1 + k1 * rad_sqr + k2 * np.square(rad_sqr))
        dx_factor = 2 * p1 * xs*ys + p2 * (rad_sqr + 2 * np.square(xs))
        xs = k_factor * (xs0 - dx_factor)
        dy_factor = 2 * p2 * xs*ys + p1 * (rad_sqr + 2 * np.square(ys))
        ys = k_factor * (ys0 - dy_factor)
    final_xs = fx * xs + cx
    final_ys = fy * ys + cy
    # we reshape array to be of form (N, 1, 2) to match OpenCV's return format
    return np.expand_dims(np.stack([final_xs, final_ys], axis=1), axis=1)


def create_view_data(img, id, feature_extractor, K, distortion):
    """ Create a view data item
    
    :param img: image read by OpenCV
    :param id: id of frame
    :param feature_extractor: OpenCV feature extractor (e.g., SIFT)
    :param K: camera matrix (shape: (3,3))
    :param distortion: distortion coefficients (list of length 4, or None if ignore distortion)
    :return: created view object
    """
    view = {
        'frame_id': id,
        'img': img,
        #'processed': False,
        'R_inC_ofW': None,
        'p_inC_ofW': None,
    }
    pts, desc = feature_extractor.detectAndCompute(image=view['img'], mask=None)
    # Undistort the points
    pts_raw = np.array([pt.pt for pt in pts])
    if distortion is not None:
        pts_final = undistort(pts_raw.copy(), K, distortion)[:,0,:]
        # Uncomment to use OpenCV undistort instead.
        # pts_final = cv2.undistortPoints(pts_raw.copy(), K, distortion, P=K)[:,0,:]
    else:
        pts_final = pts_raw.copy()
    
    view['pts'] = [
        {
            'pt2d': pts_final[i,:],
            'pt2d_raw': pts_raw[i,:],
            'track': None,
        } for i in range(pts_final.shape[0])
    ]
    view['desc'] = desc
    return view


def vo_2view(views, matching_threshold, K, R_inC_ofW, p_inC_ofW, rng,
             use_opencv=False, verbose=True,
             ransac_threshold=2e-3, ransac_iter=1000):
    """ Perform two-view reconstruction for visual odometry
    
    :param views: views to use (should be list of length 2)
    :param matching_threshold: threshold for matching
    :param K: camera matrix (shape: (3,3))
        - note: we assume views have already performed undistortion of points
    :param R_inC_ofW: rotation of world frame expressed in camera frame (shape: (3,3))
    :param p_inC_ofW: translation of world frame expressed in camera frame (shape: (3,))
    :param rng: random number generator
    :param use_opencv: if True, we use OpenCV's implementation
    :param verbose: if True, print log messages more verbosely.
    :param ransac_threshold: threshold for ransac inliers (only relevant if use_opencv=False)
    :param ransac_iter: number of ransac iterations (only relevant if use_opencv=False)
    :return: tracks
    """

    assert(len(views) == 2)

    # Get feature matches
    matches = sfm.get_good_matches(views[0]['desc'], views[1]['desc'], threshold=matching_threshold)
    if verbose:
        print(f'found {len(matches)} good matches')

    # Setup tracks
    tracks = []
    for match in matches:
        track = {
            'p_inW': None,
            'valid': True,
            'matches': [
                {'view_id': 0, 'feature_id': match.queryIdx},
                {'view_id': 1, 'feature_id': match.trainIdx},
            ]
        }
        tracks.append(track)
        views[0]['pts'][match.queryIdx]['track'] = track
        views[1]['pts'][match.trainIdx]['track'] = track

    # Create a and b (2d points for correspondences in the two images)
    a = np.array([views[0]['pts'][m.queryIdx]['pt2d'] for m in matches])
    b = np.array([views[1]['pts'][m.trainIdx]['pt2d'] for m in matches])

    if use_opencv:
        # Get 2-view solution from OpenCV
        num_inliers, E, R_inY_ofX, p_inY_ofX, mask = cv2.recoverPose(
            a.copy(),
            b.copy(),
            K, np.zeros(4),
            K, np.zeros(4),
        )
        # Flatten the position (as OpenCV returns 2D array)
        p_inY_ofX = p_inY_ofX.flatten()
        # Triangulate the points
        p_inX = cv2.triangulatePoints(
            K @ np.column_stack([np.eye(3), np.zeros(3)]), K @ np.column_stack([R_inY_ofX, p_inY_ofX]), a.copy().T, b.copy().T,
        )
        # Get p_inW (as non-homogenous points)
        p_inX = (p_inX / p_inX[-1, :])[0:3, :].T
    else:
        # Estimate essential matrix
        E, num_inliers, mask = sfm.getE(a, b, K, rng, threshold=ransac_threshold, num_iters=ransac_iter)
        if verbose:
            print(f'found {num_inliers} inliers')
        # Decompose essential matrix to estimate pose and to triangulate points
        R_inY_ofX, p_inY_ofX, p_inX = sfm.decomposeE(a, b, K, E)

    # Store pose estimates
    views[0]['R_inC_ofW'] = R_inC_ofW
    views[0]['p_inC_ofW'] = p_inC_ofW
    #views[0]['processed'] = True
    views[1]['R_inC_ofW'] = R_inY_ofX @ R_inC_ofW
    views[1]['p_inC_ofW'] = R_inY_ofX @ p_inC_ofW + p_inY_ofX
    #views[1]['processed'] = True

    # Always make sure zipped lists are the same length
    assert(len(tracks) == len(p_inX))

    # Store the position of the point corresponding to each track
    for track, p_inX_i in zip(tracks, p_inX):
        track['p_inW'] = R_inC_ofW.T @ (p_inX_i - p_inC_ofW)

    return tracks


def show_reproj_results(views, tracks, K, distortion, print_raw_reproj=True, show_reproj_histogram=True):
    """ Show the results (errors) for reprojection

    :param views: views to use
    :param tracks: tracks to use
    :param K: camera matrix (shape: (3,3))
    :param distortion: distortion coefficients (array of length 4)
    :param print_raw_reproj: if True, we print raw predictions using distortion model
    :param show_reproj_histogram: if True, we show histogram
    """
    
    # Get reprojection errors
    e_undistorted = [[] for view in views] # reprojection errors with respect to undistorted image points
    e_raw = [[] for view in views] # reprojection errors for raw detections when distortion applied during projection
    for track in tracks:
        if not track['valid']:
            continue
        
        for match in track['matches']:
            view_id = match['view_id']
            feature_id = match['feature_id']
            view = views[view_id]
            e_undistorted[view_id].append(
                sfm.projection_error(K, view['R_inC_ofW'], view['p_inC_ofW'], track['p_inW'], view['pts'][feature_id]['pt2d'], distortion=None)
            )
            e_raw[view_id].append(
                sfm.projection_error(K, view['R_inC_ofW'], view['p_inC_ofW'], track['p_inW'], view['pts'][feature_id]['pt2d_raw'], distortion=distortion)
            )
    
    print('\nREPROJECTION ERRORS')

    # Text
    for i_view, (e_undist_i, e_raw_i, view) in enumerate(zip(e_undistorted, e_raw, views)):
        if len(e_undist_i) == 0:
            assert((view['R_inC_ofW'] is None) or (view['p_inC_ofW'] is None))
            continue

        assert(not ((view['R_inC_ofW'] is None) or (view['p_inC_ofW'] is None)))
        print(f' Image {i_view:2d} ({len(e_undist_i):5d} points) : (mean, std, max, min) =' + \
                f' ({np.mean(e_undist_i):6.2f}, {np.std(e_undist_i):6.2f}, {np.max(e_undist_i):6.2f}, {np.min(e_undist_i):6.2f})')
        if print_raw_reproj:
            print(f' Image (raw reprojection) {i_view:2d} ({len(e_raw_i):5d} points) : (mean, std, max, min) =' + \
                f' ({np.mean(e_raw_i):6.2f}, {np.std(e_raw_i):6.2f}, {np.max(e_raw_i):6.2f}, {np.min(e_raw_i):6.2f})')
    
    if show_reproj_histogram:
        # Figure
        bins = np.linspace(0, 5, 50)
        counts = [len(e_i) for e_i in e_undistorted if len(e_i) > 0]
        max_count = np.max(counts)
        num_views = len(counts)
        num_cols = 3
        num_rows = (num_views // num_cols) + 1
        fig = plt.figure(figsize=(num_cols * 4, num_rows * 2), tight_layout=True)
        index = 0
        for i_view, e_i in enumerate(e_undistorted):
            if len(e_i) == 0:
                continue

            index += 1
            ax = fig.add_subplot(num_rows, num_cols, index)
            ax.hist(e_i, bins, label=f'Image {i_view}')
            ax.set_xlim([bins[0], bins[-1]])
            ax.set_ylim([0, max_count])
            ax.legend()
            ax.grid()
        plt.show()


def visualize_predictions(views, tracks, K, distortion):
    """ Visualize the results by reprojecting on the original images

    :param views: views to use
    :param tracks: tracks to use
    :param K: camera matrix (shape: (3,3))
    :param distortion: distortion coefficients (array of length 4)
    """

    # Extract the views that have been processed (non-null poses)
    selected_views = [view for view in views if view['R_inC_ofW'] is not None]
    assert(len(selected_views) == len(views)) # make sure all views selected

    # Create figure
    axs = plt.figure(figsize=(4*len(selected_views), 4)).subplots(1, len(selected_views))

    # Show each of the extracted views
    for i in range(len(selected_views)):
        view = selected_views[i]
        axs[i].imshow(selected_views[i]['img'], cmap='gray')
        axs[i].axis('off')

    # Iterate through and project points in each view as red crosses
    for t in tracks:
        if t['valid']:
            for m in t['matches']:
                R, p = views[m['view_id']]['R_inC_ofW'], views[m['view_id']]['p_inC_ofW']
                proj = sfm.project(K, R, p, np.row_stack([t['p_inW']]), warn=False, distortion=distortion)[0]
                axs[m['view_id']].plot(
                    [proj[0]],
                    [proj[1]],
                    'rx', markersize=4
                )

    # Show groundtruth points as blue crosses.
    for t in tracks:
        if t['valid']:
            for m in t['matches']:
                axs[m['view_id']].plot(
                    [views[m['view_id']]['pts'][m['feature_id']]['pt2d_raw'][0]],
                    [views[m['view_id']]['pts'][m['feature_id']]['pt2d_raw'][1]],
                    'bx', markersize=2
                )
    plt.show()


def copy_results(views, tracks):
    """
    Returns a deep copy of views and tracks so that you can store intermediate results.
    """
    # Copy views (except for references to tracks)
    views_copy = []
    for view in views:
        # Copy the view
        view_copy = {
            'frame_id': view['frame_id'],
            'img': view['img'].copy(),
            'R_inC_ofW': None if view['R_inC_ofW'] is None else view['R_inC_ofW'].copy(),
            'p_inC_ofW': None if view['p_inC_ofW'] is None else view['p_inC_ofW'].copy(),
            'pts': [],
            'desc': view['desc'].copy(),
        }

        # Copy all points in the view
        for pt in view['pts']:
            view_copy['pts'].append({
                'pt2d': pt['pt2d'].copy(),
                'pt2d_raw': pt['pt2d_raw'].copy(),
                'track': None,
            })
        
        # Append view copy to list of views
        views_copy.append(view_copy)

    # Copy tracks
    tracks_copy = []
    for track in tracks:
        # Copy the track
        track_copy = {
            'p_inW': None if track['p_inW'] is None else track['p_inW'].copy(),
            'valid': track['valid'],
            'matches': [],
        }

        # Copy all matches in the track
        for match in track['matches']:
            track_copy['matches'].append({
                'view_id': match['view_id'],
                'feature_id': match['feature_id'],
            })
        
        # Append track copy to list of tracks
        tracks_copy.append(track_copy)
    
    # Insert references to tracks into views
    for track in tracks_copy:
        for match in track['matches']:
            pt = views_copy[match['view_id']]['pts'][match['feature_id']]
            if pt['track'] is None:
                pt['track'] = track
            else:
                assert(pt['track'] is track)
    
    return views_copy, tracks_copy


def store_results(views, tracks, K, result, max_reprojection_err=1.):
    """
    Updates views and tracks given the result from optimization.
    """

    # Get pose estimates
    num_views = 0
    for i_view, view in enumerate(views):
        if (view['R_inC_ofW'] is None) or (view['p_inC_ofW'] is None):
            continue

        T_inC_ofW = result.optimized_values[f'T_inC{i_view}_ofW'].to_homogenous_matrix()
        R_inC_ofW = T_inC_ofW[0:3, 0:3]
        p_inC_ofW = T_inC_ofW[0:3, 3]
        view['R_inC_ofW'] = R_inC_ofW
        view['p_inC_ofW'] = p_inC_ofW
        num_views += 1

    # Get position estimates
    num_invalid_old = 0
    num_invalid_new = 0
    num_valid = 0
    for i_track, track in enumerate(tracks):
        if not track['valid']:
            num_invalid_old += 1
            continue
        
        p_inW = result.optimized_values[f'track_{i_track}_p_inW']
        track['p_inW'] = p_inW
        valid = track['valid']
        for match in track['matches']:
            view_id = match['view_id']
            feature_id = match['feature_id']
            view = views[view_id]
            R_inC_ofW = view['R_inC_ofW']
            p_inC_ofW = view['p_inC_ofW']
            p_inC = R_inC_ofW @ p_inW + p_inC_ofW
            b = views[view_id]['pts'][feature_id]['pt2d']
            e = sfm.projection_error(K, R_inC_ofW, p_inC_ofW, p_inW, b)
            
            # Remain valid if depth is positive
            valid = valid and p_inC[2] > 0.
            
            # Remain valid if reprojection error is below threshold
            valid = valid and e < max_reprojection_err
        
        track['valid'] = valid
        if valid:
            num_valid += 1
        else:
            num_invalid_new += 1

    # Show diagnostics
    #print(f'{num_views:6d} views with updated pose estimate')
    #print(f'{num_valid:6d} valid tracks with updated position estimate')
    #print(f'{num_invalid_old:6d} already invalid tracks')
    #print(f'{num_invalid_new:6d} newly invalid tracks')


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
        T_ofCi_inW: sf.Pose3,
        vel_i: sf.V3,
        T_ofCj_inW: sf.Pose3,
        vel_j: sf.V3,
        # acc_bias_i: sf.V3,
        # gyr_bias_i: sf.V3,
        # IMU preintegration results
        dR_ij: sf.Matrix33,
        dv_ij: sf.V3,
        dp_ij: sf.V3,
        sqrt_info_ij: sf.Matrix99,
        dt_ij: sf.Scalar,
        # Constants
        gravity: sf.V3,
        T_ofB_inC: sf.Pose3,
        # Epsilon
        epsilon: sf.Scalar,
        ) -> sf.V9:
    T_ofBi_inW = T_ofB_inC * T_ofCi_inW
    T_ofBj_inW = T_ofB_inC * T_ofCj_inW

    R_i = T_ofBi_inW.R.to_rotation_matrix()
    p_i = T_ofBi_inW.t
    R_j = T_ofBj_inW.R.to_rotation_matrix()
    p_j = T_ofBj_inW.t

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
def get_optimizer(views, tracks, acc_meas, gyr_meas, K, T_inC_ofB, 
                  vel_0=None, vel_1=None, T_C0_W=None, T_C1_W=None):
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
        if (view['R_inC_ofW'] is None) or (view['p_inC_ofW'] is None):
            continue
        
        initial_values[f'T_inC{i}_ofW'] = sym.Pose3(
            R=sym.Rot3.from_rotation_matrix(view['R_inC_ofW']),
            t=view['p_inC_ofW'],
        )

        if i > 0:
            optimized_keys.append(f'T_inC{i}_ofW')
            #print(f' T_inC{i}_ofW has an initial value and is an optimized key')
        #else:
        #    print(f' T_inC{i}_ofW has an initial value')


    if not (T_C0_W is None and T_C1_W is None):
        initial_values['T_inC0_ofW'] = sym.Pose3(
            R=sym.Rot3.from_rotation_matrix(T_C0_W[:3,:3]),
            t=T_C0_W[:3,-1],
        )
        initial_values['T_inC1_ofW'] = sym.Pose3(
            R=sym.Rot3.from_rotation_matrix(T_C1_W[:3,:3]),
            t=T_C1_W[:3,-1],
        )

    # Add a factor to fix the scale (the relative distance between frames B0 and
    # B1 will be something close to one).
    ### NOTE: Below is commented out for IMU setup
    #print(f'T_inC{1}_ofW has an sf_scale_residual factor')
    # factors = [
    #     Factor(
    #         residual=sfm.sf_scale_residual,
    #         keys=[
    #             f'T_inC{1}_ofW',
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
        #    print(f'  track_{i_track}_p_inW has an initial value and is an optimized key')
        #elif (i_track == 1):
        #    print('\n ...\n')
        initial_values[f'track_{i_track}_p_inW'] = track['p_inW']
        optimized_keys.append(f'track_{i_track}_p_inW')

        for match in track['matches']:
            view_id = match['view_id']
            feature_id = match['feature_id']
            #if (i_track == 0) or (i_track == len(tracks) - 1):
            #    print(f'  track_{i_track}_b_{view_id} has an initial value and an sf_projection_residual factor')
            initial_values[f'track_{i_track}_b_{view_id}'] = views[view_id]['pts'][feature_id]['pt2d']
            factors.append(Factor(
                residual=sfm.sf_projection_residual,
                keys=[
                    f'T_inC{view_id}_ofW',
                    f'track_{i_track}_p_inW',
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

    if not (vel_0 is None and vel_1 is None):
        initial_values['vel_0'] = vel_0
        initial_values['vel_1'] = vel_1

    # Below are values remaining constant
    assert(len(views) == 2)
    for i in range(len(views)-1):
        initial_values[f'dR_{i}{i+1}'] = dR
        initial_values[f'dv_{i}{i+1}'] = dv
        initial_values[f'dp_{i}{i+1}'] = dp
        initial_values[f'sqrt_info_{i}{i+1}'] = sqrt_info
        initial_values[f'dt_{i}{i+1}'] = dt * len(acc_meas)
    
    initial_values['gravity'] = np.array([0., 0., -9.81])
    initial_values['T_inB_ofC'] = sym.Pose3(R=sym.Rot3.from_rotation_matrix(T_inC_ofB[:3,:3].T), 
                                            t=-T_inC_ofB[:3,:3].T @ T_inC_ofB[:3,-1])
    initial_values['Cov_acc_bias'] = Cov_acc_bias
    initial_values['Cov_gyr_bias'] = Cov_gyr_bias

    # IMPORTANT: Rescale translation between views to be at scale
    if (T_C0_W is None and T_C1_W is None):
        dp = initial_values['T_inC0_ofW'].R.to_rotation_matrix().T @ T_inC_ofB[:3,:3] @ initial_values['dp_01'] \
            + initial_values['vel_0'] * initial_values['dt_01'] + 0.5 * initial_values['gravity'] * initial_values['dt_01'] ** 2
        # print(np.linalg.norm(dp))
        # print(np.linalg.norm(initial_values['T_inC1_ofW'].t))
        initial_values['T_inC1_ofW'] = sym.Pose3(R=initial_values['T_inC1_ofW'].R, 
                                                 t=np.linalg.norm(dp) / np.linalg.norm(initial_values['T_inC1_ofW'].t) * initial_values['T_inC1_ofW'].t)

    # Step 3-1: Add a residual associating poses and velocities between frames
    factors.append(Factor(
                residual=imu_preintegration_residual,
                keys=[
                    'T_inC0_ofW', 'vel_0',
                    'T_inC1_ofW', 'vel_1',
                    'dR_01',
                    'dv_01',
                    'dp_01',
                    'sqrt_info_01',
                    'dt_01',
                    'gravity',
                    'T_inB_ofC',
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


def vio_nonlinear_optimize(views, tracks, acc_meas, gyr_meas, K, T_inC_ofB, max_reprojection_err,
                           vel_0=None, vel_1=None, T_C0_W=None, T_C1_W=None):
    """ Perform non-linear optimization on views and tracks
    
    :param views: views to optimize over
    :param tracks: tracks to optimizer
    :param acc_meas: accelerometer measurements
    :param gyr_meas: gyroscope measurements
    :param K: camera matrix (shape: (3,3))
    :param T_inC_ofB: transformation from imu to camera frame (shape: (4,4))
    :param max_reprojection_err: maximum reprojection error
    :param vel_0: velocity at frame 0
    :param vel_1: velocity at frame 1
    :param vel_0: transformation matrix of frame 0 w.r.t. world frame
    :param vel_1: transformation matrix of frame 1 w.r.t. world frame
    :returns: views, tracks
    """
    optimizer, initial_values = get_optimizer(views, tracks, acc_meas, gyr_meas, K, T_inC_ofB, 
                                              vel_0, vel_1, T_C0_W, T_C1_W)
    result = optimizer.optimize(initial_values)
    new_views, new_tracks = copy_results(views, tracks)
    assert(result.status == symforce.opt.optimizer.Optimizer.Status.SUCCESS)
    # Modifies views and tracks in-place
    store_results(new_views, new_tracks, K, result, max_reprojection_err=max_reprojection_err)
    return new_views, new_tracks, initial_values, result