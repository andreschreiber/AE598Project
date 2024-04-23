import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
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


"""
Functions for two-view reconstruction.
"""

def getE(a, b, K, rng, threshold=1e-3, num_iters=1000):
    """
    INPUTS
    
    -- required --
    a is n x 2
    b is n x 2
    K is 3 x 3
    rng is a random number generator
    
    -- optional --
    threshold is the max error (e.g., epipolar or sampson) for inliers
    num_iters is the number of RANSAC iterations
    
    OUTPUTS

    E is 3 x 3
    num_inliers is a scalar
    mask is length n (truthy value for each match that is an inlier, falsy otherwise)
    """
    # Get inverse of K
    Kinv = np.linalg.inv(K)
    # Compute alpha and beta
    a_h = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    b_h = np.concatenate([b, np.ones((b.shape[0], 1))], axis=1)
    alpha = np.row_stack([Kinv @ a_h[i] for i in range(a_h.shape[0])])
    beta = np.row_stack([Kinv @ b_h[i] for i in range(b_h.shape[0])])

    # Estimate essential matrix with RANSAC
    # Essentially, we randomly sample subsets (of size 8) of points and fit on those subsets, then measure
    # epipolar distances for all points (not just the subset). The solution (of all iterations) with the greatest number
    # points that have epipolar distances less than the inlier_threshold will be our final solution (if multiple solutions
    # had the best number of inliers, the one with the smallest inlier error is chosen).
    Ebest = None
    num_inliers_best = 0
    inlier_error_best = float('inf')
    inlier_mask_best = None

    for n in range(num_iters):
        # Randomly sample 8 indices of data to fit to
        sample_indices = rng.choice(list(range(0, a.shape[0])), 8, replace=False)

        # Estimate essential matrix using the homogenous system (the one with a matrix of Kronecker products given in the theory section of the notebook)
        q1 = alpha[sample_indices]
        q2 = beta[sample_indices]
        M = np.array([
            [
                q2[i][0]*q1[i][0], q2[i][0]*q1[i][1], q2[i][0]*q1[i][2],
                q2[i][1]*q1[i][0], q2[i][1]*q1[i][1], q2[i][1]*q1[i][2],
                q2[i][2]*q1[i][0], q2[i][2]*q1[i][1], q2[i][2]*q1[i][2],
            ] for i in range(q1.shape[0])
        ])
        # We can solve the system using SVD as described in the markdown cell, and then reshape to a 3x3 matrix.
        Epp = np.linalg.svd(M)[-1][-1].reshape((3,3))

        # Normalize essential matrix so p_inB_ofA has norm 1
        Ep = (np.sqrt(2) / np.linalg.norm(Epp)) * Epp

        # Correct the essential matrix as described in theory section of notebook
        Up, Sp, VpT = np.linalg.svd(Ep)
        Vp = VpT.T
        U = np.stack([Up[:,0], Up[:,1], Up[:,2] * np.linalg.det(Up)], axis=1)
        V = np.stack([Vp[:,0], Vp[:,1], Vp[:,2] * np.linalg.det(Vp)], axis=1)
        S = np.diag(np.array([1.0, 1.0, 0.0]))
        E = U @ S @ V.T

        # Now, we calculate epipolar distances so we can figure out the number of inliers.
        # Numerator for epipolar distance is same for both d_Bi and d_Ai
        epipolar_numerator = np.array([
            beta[i].T @ E @ alpha[i] for i in range(alpha.shape[0])
        ])
        # Compute denominators for expression of d_Bi
        epipolar_denominator_Bi = np.array([
            np.linalg.norm((skew(np.array([0.0, 0.0, 1.0])) @ E) @ alpha[i]) for i in range(alpha.shape[0])
        ])
        # Compute denominators for expression of d_Ai
        epipolar_denominator_Ai = np.array([
            np.linalg.norm((skew(np.array([0.0, 0.0, 1.0])) @ E.T) @ beta[i]) for i in range(beta.shape[0])
        ])

        # Compute epipolar distances
        d_Bi = np.abs(epipolar_numerator / epipolar_denominator_Bi)
        d_Ai = np.abs(epipolar_numerator / epipolar_denominator_Ai)
        # Compute number of inliers
        inliers = np.logical_and(d_Ai < threshold, d_Bi < threshold)
        num_inliers = inliers.sum()
        inlier_error = d_Bi[inliers].sum() + d_Ai[inliers].sum()
        # Check if we have a new best solution
        if num_inliers > num_inliers_best or (num_inliers == num_inliers_best and inlier_error < inlier_error_best):
            # We have a new best solution!
            inlier_error_best = inlier_error
            num_inliers_best = num_inliers
            inlier_mask_best = inliers
            Ebest = E
    
    return Ebest, num_inliers_best, inlier_mask_best

def twoview_triangulate(alpha, beta, R_inB_ofA, p_inB_ofA):
    # Two-view triangulate from HW2 (used to help with decomposing E)
    # INPUTS (alpha, beta, R_inB_ofA, p_inB_ofA)
    #  alpha        normalized coordinates of points in image A
    #  beta         normalized coordinates of points in image B
    #  R_inB_ofA    orientation of frame A in frame B
    #  p_inB_ofA    position of frame A in frame B
    #
    # OUTPUTS (p_inA, p_inB, mask)
    #  p_inA        triangulated points in frame A
    #  p_inB        triangulated points in frame B
    #  mask         1d array of length equal to number of triangulated points,
    #               with a "1" for each point that has positive depth in both
    #               frames and with a "0" otherwise

    # Initialize points
    p_inA = []
    p_inB = []

    # Loop through the points for triangulation
    for i in range(len(alpha)):
        # Apply procedure described in above markup cell to triangulate
        alphai, betai = alpha[i], beta[i]
        # Compute wedge version of alpha_i and beta_i
        salphai, sbetai = skew(alphai), skew(betai)
        # Solve the linear system for triangulation described in the above markdown cell (equation has form Ax=B, where x = p_inA_i)
        # Construct least squares A matrix
        lstqA = np.concatenate([np.dot(sbetai, R_inB_ofA), salphai], axis=0)
        # Construct least squares B vector
        lstqB = np.concatenate([-np.dot(sbetai, p_inB_ofA), np.zeros(3)], axis=0)
        # Get least squares solution (which is p_inA)
        lstqSoln = np.linalg.lstsq(lstqA, lstqB, rcond=None)[0]
        p_inA.append(lstqSoln)
        # Get p_inB via coordinate transform
        p_inB.append(np.dot(R_inB_ofA, lstqSoln) + p_inB_ofA)
    p_inA = np.array(p_inA)
    p_inB = np.array(p_inB)
    # Mask should be 1 if point has positive depth in both frames
    mask = np.logical_and(p_inA[:,2] > 0, p_inB[:,2] > 0)
    return (p_inA, p_inB, mask)

def get_possibility(U, V, index=0):
    # Get a possible decomposition of the essential matrix (where U and V are matrices of left and right singular vectors)
    # INPUTS (U, V, index)
    #   U       left singular vectors of E
    #   V       right singular vectors of E
    #   index   index of possibility (integer from 0 to 3, as there are 4 possible decompositions)
    # Outputs
    #   (R, p) where R is a rotation matrix and p is a translation

    # Create W matrix for decomposition
    W = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])

    # Return one of the possible decompositions according to the index (as described in theory section of notebook)
    if index == 0:
        return (U @ W.T @ V.T,  U[:,2])
    elif index == 1:
        return (U @ W @ V.T,   -U[:,2])
    elif index == 2:
        return (U @ W.T @ V.T, -U[:,2])
    elif index == 3:
        return (U @ W @ V.T,    U[:,2])
    else:
        raise ValueError("Invalid index")

def decomposeE(a, b, K, E):
    """
    INPUTS
    
    -- required --
    a is n x 2
    b is n x 2
    K is 3 x 3
    E is 3 x 3
    
    OUTPUTS

    R_inB_ofA is 3 x 3
    p_inB_ofA is length 3
    p_inA is n x 3
    """
    # Get inverse of K
    Kinv = np.linalg.inv(K)
    # Compute alpha and beta
    a_h = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    b_h = np.concatenate([b, np.ones((b.shape[0], 1))], axis=1)
    alpha = np.row_stack([Kinv @ a_h[i] for i in range(a_h.shape[0])])
    beta = np.row_stack([Kinv @ b_h[i] for i in range(b_h.shape[0])])

    # Get U and V
    Up, Sp, VpT = np.linalg.svd(E)
    Vp = VpT.T
    # Make sure U and V satisfy the properties requires (note: the singular value
    # is 0, so I apply the determinant correction just to be safe)
    U = np.stack([Up[:,0], Up[:,1], Up[:,2] * np.linalg.det(Up)], axis=1)
    V = np.stack([Vp[:,0], Vp[:,1], Vp[:,2] * np.linalg.det(Vp)], axis=1)

    # Decompose essential matrix and do triangulation
    finalR, finalP = None, None
    p_inA, p_inB = None, None
    num_valid = -1 # number of "valid" points (points with positive depth)
    for i in range(4): # Test each of the four decomposition possibilities
        R, p = get_possibility(U, V, index=i)
        pA, pB, mask = twoview_triangulate(alpha, beta, R, p)
        cur_num_valid = mask.sum()
        if cur_num_valid > num_valid: # Update if we have fewer negative depths than prior best
            finalR, finalP = R, p
            p_inA, p_inB = pA, pB
            num_valid = cur_num_valid

    return finalR, finalP, p_inA


"""
Functions for resectioning and triangulation.
"""

def resection(p_inA, c, K, rng, threshold=1., num_iters=1000, n_fit=10):
    """
    INPUTS
    
    -- required --
    p_inA is n x 3
    c is n x 2
    K is 3 x 3
    rng is a random number generator
    
    -- optional --
    threshold is the max error (e.g., reprojection) for inliers
    num_iters is the number of RANSAC iterations
    
    OUTPUTS

    R_inC_ofA is 3 x 3
    p_inC_ofA is length 3
    num_inliers is a scalar
    mask is length n (truthy value for each match that is an inlier, falsy otherwise)
    """

    # Get inverse of K
    Kinv = np.linalg.inv(K)
    # Compute gammas
    c_h = np.concatenate([c, np.ones((c.shape[0], 1))], axis=1)
    gamma = np.row_stack([Kinv @ c_h[i] for i in range(c_h.shape[0])])

    # Resection using RANSAC
    # Here, we perform resectioning where we fit to subsets of the data, and then
    # measure reprojection error on all data (not just data we used for fitting).
    # The solution with the highest number of inliers is taken as our final solution
    # (if multiple "best" solutions exist by having the same number of inliers, we take
    # the one that has lowest inlier error).
    R_inC_ofA_best, p_inC_ofA_best = None, None
    num_inliers_best = 0
    inlier_error_best = float('inf')
    inlier_mask_best = None

    for n in range(num_iters):
        # Minimal set to fit is of size 6, but can use more for higher accuracy.
        sample_indices = rng.choice(list(range(0, c.shape[0])), n_fit, replace=False)
        # Create matrix of kronecker products and find solution via SVD (see theory section of notebook for details)
        M = np.row_stack([np.concatenate([np.kron(p_inA[i], skew(gamma[i])), skew(gamma[i])], axis=1) for i in sample_indices])
        U, S, VT = np.linalg.svd(M)
        # Extract solution
        soln = np.array([
            [VT[-1][0], VT[-1][3], VT[-1][6], VT[-1][9]],
            [VT[-1][1], VT[-1][4], VT[-1][7], VT[-1][10]],
            [VT[-1][2], VT[-1][5], VT[-1][8], VT[-1][11]]
        ])
        # Correct the solution
        # Should have the right scale
        soln = soln / np.linalg.norm(soln[:,0])
        # Should be right-handed
        soln = soln * np.linalg.det(soln[:3,:3])
        # R_inC_ofA should be a rotation matrix
        U, S, VT = np.linalg.svd(soln[:3,:3])
        R_inC_ofA = np.linalg.det(U @ VT) * (U @ VT)
        p_inC_ofA = soln[:,3]

        # Compute projection errors
        reprojection_errors = projection_error(K, R_inC_ofA, p_inC_ofA, p_inA, c, warn=False)
        # Compute number of inliers
        inliers = reprojection_errors < threshold
        num_inliers = inliers.sum()
        inlier_error = reprojection_errors[inliers].sum()
        if num_inliers > num_inliers_best or (num_inliers == num_inliers_best and inlier_error < inlier_error_best):
            # New best solution!
            inlier_error_best = inlier_error
            num_inliers_best = num_inliers
            inlier_mask_best = inliers
            R_inC_ofA_best = R_inC_ofA
            p_inC_ofA_best = p_inC_ofA

    return R_inC_ofA_best, p_inC_ofA_best, num_inliers_best, inlier_mask_best

def triangulate(track, views, K):
    """
    INPUTS
    track is **one** track of matches to triangulate
    views is the list of all views
    K is 3 x 3
    
    OUTPUTS

    p_inA is length 3
    """
    # Get inverse of K
    Kinv = np.linalg.inv(K)

    # We create the matrix equation for triangulation (of form Ax=b, where x = p_inA) (see appendix of notebook for derivation)
    lstq_A = []
    lstq_b = []
    for match in track['matches']:
        view_id = match['view_id']
        feature_id = match['feature_id']
        betai = Kinv @ np.append(views[view_id]['pts'][feature_id]['pt2d'], 1.0)
        sbetai = skew(betai)

        lstq_A.append( sbetai @ views[view_id]['R_inB_ofA'])
        lstq_b.append(-sbetai @ views[view_id]['p_inB_ofA'])

    # Concatenate to make np arrays (to prepare for solving)
    lstq_A = np.concatenate(lstq_A, axis=0)
    lstq_b = np.concatenate(lstq_b, axis=0)

    # Solve system for triangulated point
    p_inA = np.linalg.lstsq(lstq_A, lstq_b, rcond=None)[0]
    return p_inA


"""
Functions for optimization
"""

def get_optimizer(views, tracks, K):
    """
    Returns a symforce optimizer and a set of initial_values that
    allow you to solve the bundle adjustment problem corresponding
    to the given views and tracks.
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
    #print(f'T_inB{1}_ofA has an sf_scale_residual factor')
    factors = [
        Factor(
            residual=sf_scale_residual,
            keys=[
                f'T_inB{1}_ofA',
                'epsilon',
            ],
        )
    ]

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
                residual=sf_projection_residual,
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




# The following code was provided by Professor Bretl and has not been modified.

def myprint(M):
    """
    Prints either a scalar or a numpy array with four digits after the
    decimal and with consistent spacing so it is easier to read.
    """
    if M.shape:
        with np.printoptions(linewidth=150, formatter={'float': lambda x: f'{x:10.4f}'}):
            print(M)
    else:
        print(f'{M:10.4f}')


def apply_transform(R_inB_ofA, p_inB_ofA, p_inA):
    """
    Returns p_inB.
    """
    p_inB = np.row_stack([
        (R_inB_ofA @ p_inA_i + p_inB_ofA) for p_inA_i in p_inA
    ])
    return p_inB

def project(K, R_inB_ofA, p_inB_ofA, p_inA, warn=False, distortion=None):
    """
    Returns the projection (n x 2) of p_inA (n x 3) into an image
    taken from a calibrated camera at frame B.

    Modified version from homework that also supports distortion.
    """
    p_inB = apply_transform(R_inB_ofA, p_inB_ofA, p_inA)
    if not np.all(p_inB[:, 2] > 0):
        if warn:
            print('WARNING: non-positive depths')
    
    if distortion is None:
        # If no distortion, use a simple projection model
        q = np.row_stack([K @ p_inB_i / p_inB_i[2] for p_inB_i in p_inB])
    else:
        # For distortion, use model from here: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
        # get intrinsics
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        # get distortion parameters
        k1, k2, p1, p2 = distortion
        # use equations to determine the projected locations
        q_p = np.row_stack([p_inB_i / p_inB_i[2] for p_inB_i in p_inB])
        x_p, y_p = q_p[:,0], q_p[:,1]
        rad_sqr = np.square(x_p) + np.square(y_p)
        x_pp = (x_p * (1 + k1 * rad_sqr + k2 * np.square(rad_sqr))) + (2*p1*x_p*y_p) + p2*(rad_sqr + 2*np.square(x_p))
        y_pp = (y_p * (1 + k1 * rad_sqr + k2 * np.square(rad_sqr))) + (2*p2*x_p*y_p) + p1*(rad_sqr + 2*np.square(y_p))
        q = np.stack([fx * x_pp + cx, fy * y_pp + cy], axis=1)
    return q[:, 0:2]

def projection_error(K, R_inB_ofA, p_inB_ofA, p_inA, b, warn=False, distortion=None):
    """
    Returns the projection error - the difference between the projection
    of p_inA into an image taken from a calibrated camera at frame B, and
    the observed image coordinates b.

    If p_inA is n x 3 and b is n x 2, then the error is length n.

    If p_inA is length 3 and b is length 2, then the error is a scalar.
    """
    if len(b.shape) == 1:
        b_pred = project(K, R_inB_ofA, p_inB_ofA, np.reshape(p_inA, (1, -1)), warn=warn, distortion=distortion).flatten()
        return np.linalg.norm(b_pred - b)
    elif len(b.shape) == 2:
        b_pred = project(K, R_inB_ofA, p_inB_ofA, p_inA, warn=warn, distortion=distortion)
        return np.linalg.norm(b_pred - b, axis=1)
    else:
        raise Exception(f'b has bad shape: {b.shape}')
    
def skew(v):
    """
    Returns the 3 x 3 skew-symmetric matrix that corresponds
    to v, a vector of length 3.
    """
    assert(type(v) == np.ndarray)
    assert(v.shape == (3,))
    return np.array([[0., -v[2], v[1]],
                     [v[2], 0., -v[0]],
                     [-v[1], v[0], 0.]])

def show_results(views, tracks, K, show_pose_estimates=True, show_reprojection_errors=True):
    """
    Show the pose estimates (text) and reprojection errors (both text and plots)
    corresponding to views and tracks.
    """

    # Pose estimates
    if show_pose_estimates:
        print('POSE ESTIMATES')
        for i_view, view in enumerate(views):
            if (view['R_inB_ofA'] is None) or (view['p_inB_ofA'] is None):
                continue

            R_inA_ofB = view['R_inB_ofA'].T
            p_inA_ofB = - view['R_inB_ofA'].T @ view['p_inB_ofA']
            s = f' [R_inA_ofB{i_view}, p_inA_ofB{i_view}] = '
            s += np.array2string(
                np.column_stack([R_inA_ofB, p_inA_ofB]),
                formatter={'float': lambda x: f'{x:10.4f}'},
                prefix=s,
            )
            print(s)
    
    # Get reprojection errors
    e = [[] for view in views]
    for track in tracks:
        if not track['valid']:
            continue
        
        for match in track['matches']:
            view_id = match['view_id']
            feature_id = match['feature_id']
            view = views[view_id]
            e[view_id].append(
                projection_error(
                    K,
                    view['R_inB_ofA'],
                    view['p_inB_ofA'],
                    track['p_inA'],
                    view['pts'][feature_id]['pt2d'],
                )
            )
    
    # Show reprojection errors
    if show_reprojection_errors:
        print('\nREPROJECTION ERRORS')

        # Text
        for i_view, (e_i, view) in enumerate(zip(e, views)):
            if len(e_i) == 0:
                assert((view['R_inB_ofA'] is None) or (view['p_inB_ofA'] is None))
                continue

            assert(not ((view['R_inB_ofA'] is None) or (view['p_inB_ofA'] is None)))
            print(f' Image {i_view:2d} ({len(e_i):5d} points) : (mean, std, max, min) =' + \
                  f' ({np.mean(e_i):6.2f}, {np.std(e_i):6.2f}, {np.max(e_i):6.2f}, {np.min(e_i):6.2f})')
        
        # Figure
        bins = np.linspace(0, 5, 50)
        counts = [len(e_i) for e_i in e if len(e_i) > 0]
        max_count = np.max(counts)
        num_views = len(counts)
        num_cols = 3
        num_rows = (num_views // num_cols) + 1
        fig = plt.figure(figsize=(num_cols * 4, num_rows * 2), tight_layout=True)
        index = 0
        for i_view, e_i in enumerate(e):
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

def is_duplicate_match(mA, matches):
    """
    Returns True if the match mA shares either a queryIdx or a trainIdx
    with any match in the list matches, False otherwise.
    """
    for mB in matches:
        if (mA.queryIdx == mB.queryIdx) or (mA.trainIdx == mB.trainIdx):
            return True
    return False

def get_good_matches(descA, descB, threshold=0.5):
    """
    Returns a list of matches that satisfy the ratio test with
    a given threshold. Makes sure these matches are unique - no
    feature in either image will be part of more than one match.
    """

    # Create a brute-force matcher
    bf = cv2.BFMatcher(
        normType=cv2.NORM_L2,
        crossCheck=False,       # <-- IMPORTANT - must be False for kNN matching
    )

    # Find the two best matches between descriptors
    matches = bf.knnMatch(descA, descB, k=2)

    # Find the subset of good matches
    good_matches = []
    for m, n in matches:
        if m.distance / n.distance < threshold:
            good_matches.append(m)

    # Sort the good matches by distance (smallest first)
    sorted_matches = sorted(good_matches, key = lambda m: m.distance)

    # VERY IMPORTANT - Eliminate duplicate matches
    unique_matches = []
    for sorted_match in sorted_matches:
        if not is_duplicate_match(sorted_match, unique_matches):
            unique_matches.append(sorted_match)
    
    # Return good matches, sorted by distance (smallest first)
    return unique_matches

def remove_track(track_to_remove, tracks):
    """
    Removes the given track from a list of tracks. (This is non-trivial
    because tracks are dictionaries for which we don't have a built-in notion
    of equality. An alternative would be to create a class for tracks, and to
    define an equality operator for that class.)
    """

    i_to_remove = [i_track for i_track, track in enumerate(tracks) if track is track_to_remove]
    assert(len(i_to_remove) == 1)
    del tracks[i_to_remove[0]]

def add_next_view(views, tracks, K, matching_threshold=0.5, lookback=-1):
    """
    Updates views and tracks after matching the next available image.
    """
    iC = None
    for i_view, view in enumerate(views):
        if (view['R_inB_ofA'] is None) or (view['p_inB_ofA'] is None):
            #if view['processed'] == False:
            iC = i_view
            break
    
    if iC is None:
        raise Exception('all views have been added')
    
    print(f'ADDING VIEW {iC}')
    
    match_start_index = 0
    if lookback > 0:
        match_start_index = max(0, iC - lookback)

    for iB in range(match_start_index, iC):

        viewB = views[iB]
        viewC = views[iC]

        # Get good matches
        matches = get_good_matches(viewB['desc'], viewC['desc'], threshold=matching_threshold)
        num_matches = len(matches)
        #print('')
        #print(f'matching image {iB} with image {iC} with threshold {matching_threshold}:')
        #print(f' {num_matches:4d} good matches found')
        
        num_created = 0
        num_added_to_B = 0
        num_added_to_C = 0
        num_merged_trivial = 0
        num_merged_nontrivial = 0
        num_invalid = 0

        for m in matches:
            # Get the corresponding points
            ptB = viewB['pts'][m.queryIdx]
            ptC = viewC['pts'][m.trainIdx]

            # Get the corresponding tracks (if they exist)
            trackB = ptB['track']
            trackC = ptC['track']

            # Create, extend, or merge tracks
            if (trackC is None) and (trackB is None):
                num_created += 1
                # Create a new track
                track = {
                    'p_inA': None,
                    'valid': True,
                    'matches': [
                        {'view_id': iB, 'feature_id': m.queryIdx},
                        {'view_id': iC, 'feature_id': m.trainIdx},
                    ],
                }
                tracks.append(track)
                ptB['track'] = track
                ptC['track'] = track
            elif (trackC is not None) and (trackB is None):
                num_added_to_C += 1
                # Add ptB to trackC
                track = trackC
                trackC['matches'].append({'view_id': iB, 'feature_id': m.queryIdx})
                ptB['track'] = track
            elif (trackC is None) and (trackB is not None):
                num_added_to_B += 1
                # Add ptC to trackB
                track = trackB
                trackB['matches'].append({'view_id': iC, 'feature_id': m.trainIdx})
                ptC['track'] = track
            elif (trackC is not None) and (trackB is not None):
                
                # If trackB and trackC are identical, then nothing further needs to be done
                if trackB is trackC:
                    num_merged_trivial += 1
                    s = f'       trivial merge - ({iB:2d}, {m.queryIdx:4d}) ({iC:2d}, {m.trainIdx:4d}) - '
                    for track_m in trackB['matches']:
                        s += f'({track_m["view_id"]:2d}, {track_m["feature_id"]:4d}) '
                    #print(s)
                    continue
                
                num_merged_nontrivial += 1

                s = f'       non-trivial merge - matches ({iB:2d}, {m.queryIdx:4d}) ({iC:2d}, {m.trainIdx:4d})\n'
                s += '                           track one '
                for track_m in trackB['matches']:
                    s += f'({track_m["view_id"]:2d}, {track_m["feature_id"]:4d}) '
                s += '\n'
                s += '                           track two '
                for track_m in trackC['matches']:
                    s += f'({track_m["view_id"]:2d}, {track_m["feature_id"]:4d}) '
                #print(s)

                # Merge
                # - triangulated point
                if trackB['p_inA'] is None:
                    if trackC['p_inA'] is None:
                        p_inA = None
                    else:
                        p_inA = trackC['p_inA']
                else:
                    if trackC['p_inA'] is None:
                        p_inA = trackB['p_inA']
                    else:
                        # FIXME: May want to re-triangulate rather than averaging
                        p_inA = 0.5 * (trackB['p_inA'] + trackC['p_inA'])
                trackC['p_inA'] = p_inA
                # - valid
                valid = trackB['valid'] and trackC['valid']
                trackC['valid'] = valid
                # - matches and points
                for trackB_m in trackB['matches']:
                    # ONLY add match to track if it isn't already there (duplicate matches
                    # can happen if we closed a loop)
                    if (trackB_m['view_id'] != iC) or (trackB_m['feature_id'] != m.trainIdx):
                        trackC['matches'].append(trackB_m)
                    
                    # ALWAYS update track of point corresponding to match
                    views[trackB_m['view_id']]['pts'][trackB_m['feature_id']]['track'] = trackC
                track = trackC

                # Remove the leftover track
                remove_track(trackB, tracks)
                
                s = '                           => '
                for track_m in track['matches']:
                    s += f'({track_m["view_id"]:2d}, {track_m["feature_id"]:4d}) '
                s += f' - {str(track["valid"])}'
                #print(s)

            else:
                raise Exception('Should never get here!')
            
            # Check if track is self-consistent (i.e., check that it does not contain two
            # matches from the same image)
            view_ids = [track_m['view_id'] for track_m in track['matches']]
            if len(set(view_ids)) != len(view_ids):
                num_invalid += 1
                track['valid'] = False
                s = f'       FOUND INCONSISTENT - '
                for track_m in track['matches']:
                    s += f'({track_m["view_id"]:2d}, {track_m["feature_id"]:4d}) '
                #print(s)

        #print(f' {num_created:4d} tracks created')
        #print(f' {num_added_to_C:4d} tracks in C extended with point in B')
        #print(f' {num_added_to_B:4d} tracks in B extended with point in C')
        #print(f' {num_merged_trivial:4d} tracks merged (trivial)')
        #print(f' {num_merged_trivial:4d} tracks merged (non-trivial)')
        #print(f' {num_invalid:4d} inconsistent tracks')
    
    return iC

def store_results(views, tracks, K, result, max_reprojection_err=1.):
    """
    Updates views and tracks given the result from optimization.
    """

    # Get pose estimates
    num_views = 0
    for i_view, view in enumerate(views):
        if (view['R_inB_ofA'] is None) or (view['p_inB_ofA'] is None):
            continue

        T_inB_ofA = result.optimized_values[f'T_inB{i_view}_ofA'].to_homogenous_matrix()
        R_inB_ofA = T_inB_ofA[0:3, 0:3]
        p_inB_ofA = T_inB_ofA[0:3, 3]
        view['R_inB_ofA'] = R_inB_ofA
        view['p_inB_ofA'] = p_inB_ofA
        num_views += 1

    # Get position estimates
    num_invalid_old = 0
    num_invalid_new = 0
    num_valid = 0
    for i_track, track in enumerate(tracks):
        if not track['valid']:
            num_invalid_old += 1
            continue
        
        p_inA = result.optimized_values[f'track_{i_track}_p_inA']
        track['p_inA'] = p_inA
        valid = track['valid']
        for match in track['matches']:
            view_id = match['view_id']
            feature_id = match['feature_id']
            view = views[view_id]
            R_inB_ofA = view['R_inB_ofA']
            p_inB_ofA = view['p_inB_ofA']
            p_inB = R_inB_ofA @ p_inA + p_inB_ofA
            b = views[view_id]['pts'][feature_id]['pt2d']
            e = projection_error(K, R_inB_ofA, p_inB_ofA, p_inA, b)
            
            # Remain valid if depth is positive
            valid = valid and p_inB[2] > 0.
            
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

def get_match_with_view_id(matches, view_id):
    """
    From a list of matches, return the one with the given view_id.
    """
    for match in matches:
        if match['view_id'] == view_id:
            return match
    return None

def get_pt2d_from_match(views, match):
    """
    Get image coordinates of the given match.
    """
    return views[match['view_id']]['pts'][match['feature_id']]['pt2d']

def sf_projection(
    T_inC_ofW: sf.Pose3,
    p_inW: sf.V3,
    fx: sf.Scalar,
    fy: sf.Scalar,
    cx: sf.Scalar,
    cy: sf.Scalar,
    epsilon: sf.Scalar,
) -> sf.V2:
    """
    Symbolic function that projects a point into an image. (If the depth
    of this point is non-positive, then the projection will be pushed far
    away from the image center.)
    """
    p_inC = T_inC_ofW * p_inW
    z = sf.Max(p_inC[2], epsilon)   # <-- if depth is non-positive, then projection
                                    #     will be pushed far away from image center
    return sf.V2(
        fx * (p_inC[0] / z) + cx,
        fy * (p_inC[1] / z) + cy,
    )

def sf_projection_residual(
    T_inC_ofW: sf.Pose3,
    p_inW: sf.V3,
    q: sf.V2,
    fx: sf.Scalar,
    fy: sf.Scalar,
    cx: sf.Scalar,
    cy: sf.Scalar,
    epsilon: sf.Scalar,  
) -> sf.V2:
    """
    Symbolic function that computes the difference between a projected point
    and an image point. This error is "whitened" so that taking its norm will
    be equivalent to applying a robust loss function (Geman-McClure).
    """
    q_proj = sf_projection(T_inC_ofW, p_inW, fx, fy, cx, cy, epsilon)
    unwhitened_residual = sf.V2(q_proj - q)
    
    noise_model = BarronNoiseModel(
        alpha=-2,
        scalar_information=1,
        x_epsilon=epsilon,
        alpha_epsilon=epsilon,
    )
    
    return noise_model.whiten_norm(unwhitened_residual)

def sf_scale_residual(
    T_inC_ofW: sf.Pose3,
    epsilon: sf.Scalar,
) -> sf.V1:
    """
    Symbolic function that computes the relative distance between two frames.
    """
    return sf.V1(T_inC_ofW.t.norm() - 1)

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
            'R_inB_ofA': None if view['R_inB_ofA'] is None else view['R_inB_ofA'].copy(),
            'p_inB_ofA': None if view['p_inB_ofA'] is None else view['p_inB_ofA'].copy(),
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
            'p_inA': None if track['p_inA'] is None else track['p_inA'].copy(),
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

def visualize_results(views, tracks, K, fps):
    """
    Visualizes results with rerun.
    """
    
    # Get index of current view
    i_view = len(views)
    for i_view, view in enumerate(views):
        if (view['R_inB_ofA'] is None) or (view ['p_inB_ofA'] is None):
            break
    i_view -= 1
    
    # Set current time
    rr.set_time_seconds('stable_time', float(views[i_view]['frame_id']) / float(fps))

    # Show triangulated points
    p_inA = np.array([track['p_inA'] for track in tracks if track['valid'] and track['p_inA'] is not None])
    rr.log(
        '/results/triangulated_points',
        rr.Points3D(p_inA),
    )

    # Show camera frames
    for i_view, view in enumerate(views):
        if (view['R_inB_ofA'] is None) or (view ['p_inB_ofA'] is None):
            continue
        
        R_inB_ofA = view['R_inB_ofA'].copy()
        p_inB_ofA = view['p_inB_ofA'].copy()
        R_inA_ofB = R_inB_ofA.T
        p_inA_ofB = -R_inB_ofA.T @ p_inB_ofA
        rr.log(
            f'/results/camera_{i_view}',
            rr.Transform3D(
                translation=p_inA_ofB,
                rotation=rr.Quaternion(xyzw=Rotation.from_matrix(R_inA_ofB).as_quat()),
            )
        )

