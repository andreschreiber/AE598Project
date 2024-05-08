# Author: Andre Schreiber
# This file contains code for visual odometry implementation

import cv2
import sfm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import symforce


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
        'R_inB_ofA': None,
        'p_inB_ofA': None,
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


def vo_2view(views, matching_threshold, K, rng, use_opencv=False, verbose=True,
             ransac_threshold=2e-3, ransac_iter=1000):
    """ Perform two-view reconstruction for visual odometry
    
    :param views: views to use (should be list of length 2)
    :param matching_threshold: threshold for matching
    :param K: camera matrix (shape: (3,3))
        - note: we assume views have already performed undistortion of points
    :param rng: random number generator
    :param use_opencv: if True, we use OpenCV's implementation (NOTE: deprecated and un-tested as of final submission)
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
            'p_inA': None,
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
        num_inliers, E, R_inB_ofA, p_inB_ofA, mask = cv2.recoverPose(
            a.copy(),
            b.copy(),
            K, np.zeros(4),
            K, np.zeros(4),
        )
        # Flatten the position (as OpenCV returns 2D array)
        p_inB_ofA = p_inB_ofA.flatten()
        # Triangulate the points
        p_inA = cv2.triangulatePoints(
            K @ np.column_stack([np.eye(3), np.zeros(3)]), K @ np.column_stack([R_inB_ofA, p_inB_ofA]), a.copy().T, b.copy().T,
        )
        # Get p_inA (as non-homogenous points)
        p_inA = (p_inA / p_inA[-1, :])[0:3, :].T
    else:
        # Estimate essential matrix
        E, num_inliers, mask = sfm.getE(a, b, K, rng, threshold=ransac_threshold, num_iters=ransac_iter)
        if verbose:
            print(f'found {num_inliers} inliers')
        # Decompose essential matrix to estimate pose and to triangulate points
        R_inB_ofA, p_inB_ofA, p_inA = sfm.decomposeE(a, b, K, E)

    # Store pose estimates
    views[0]['R_inB_ofA'] = np.eye(3)
    views[0]['p_inB_ofA'] = np.zeros(3)
    #views[0]['processed'] = True
    views[1]['R_inB_ofA'] = R_inB_ofA
    views[1]['p_inB_ofA'] = p_inB_ofA
    #views[1]['processed'] = True

    # Always make sure zipped lists are the same length
    assert(len(tracks) == len(p_inA))

    # Store the position of the point corresponding to each track
    for track, p_inA_i in zip(tracks, p_inA):
        track['p_inA'] = p_inA_i

    return tracks



def vo_resection(views, tracks, matching_threshold, K, rng, use_opencv=False, verbose=True,
                 ransac_threshold=2, ransac_iter=1000):
    """ Perform resection for visual odometry
    
    :param views: views to use
    :param tracks: tracks to use
    :param matching_threshold: threshold for matching
    :param K: camera matrix (shape: (3,3))
        - note: we assume views have already performed undistortion of points
    :param rng: random number generator
    :param use_opencv: if True, we use OpenCV's implementation (NOTE: deprecated and un-tested as of final submission)
    :param verbose: if True, print log messages more verbosely.
    :param ransac_threshold: threshold for ransac inliers (only relevant if use_opencv=False)
    :param ransac_iter: number of ransac iterations (only relevant if use_opencv=False)
    :return: views, tracks
    """

    # Get the next view added
    iC = sfm.add_next_view(views, tracks, K, matching_threshold=matching_threshold)

    # Look for resectioning and triangulation tracks
    tracks_to_resection = []
    tracks_to_triangulate = []
    for track in tracks:
        if not track['valid']:
            continue
        
        match = sfm.get_match_with_view_id(track['matches'], iC)
        if match is None:
            continue

        if track['p_inA'] is None:
            tracks_to_triangulate.append(track)
        else:
            tracks_to_resection.append(track)

    if verbose:
        print(f'{len(tracks_to_resection)} tracks to resection')
        print(f'{len(tracks_to_triangulate)} tracks to triangulate')

    p_inA = [] # 3D "world" points
    c = [] # points in image
    # Get the 3D locations and projections of observed interest points (for resectioning)
    for track in tracks_to_resection:
        assert(track['p_inA'] is not None)
        p_inA.append(track['p_inA'])
        match = sfm.get_match_with_view_id(track['matches'], iC)
        c.append(sfm.get_pt2d_from_match(views, match))
    # Convert to numpy arrays
    p_inA = np.array(p_inA)
    c = np.array(c)

    if verbose:
        print(f'Resectioning: len(p_inA) = {len(p_inA)}, len(c) = {len(c)}')

    # Perform resectioning
    if use_opencv:
        retval, rvec, tvec = cv2.solvePnP(p_inA, c.copy(), K, np.zeros(4))
        assert(retval)
        R_inC_ofA = Rotation.from_rotvec(rvec.flatten()).as_matrix()
        p_inC_ofA = tvec.flatten()
    else:
        R_inC_ofA, p_inC_ofA, num_inliers, mask = sfm.resection(
            p_inA,
            c,
            K,
            rng,
            threshold=ransac_threshold,
            num_iters=ransac_iter,
        )
        if verbose:
            print(f'found {num_inliers} inliers out of {len(mask)}')

    # Store resectioning results
    views[iC]['R_inB_ofA'] = R_inC_ofA
    views[iC]['p_inB_ofA'] = p_inC_ofA
    #views[iC]['processed'] = True

    # Perform triangulation
    for track in tracks_to_triangulate:
        p_inA = sfm.triangulate(track, views, K)
        track['p_inA'] = p_inA

    return views, tracks


def vo_nonlinear_optimize(views, tracks, K, max_reprojection_err):
    """ Perform non-linear optimization on views and tracks
    
    :param views: views to optimize over
    :param tracks: tracks to optimizer
    :param K: camera matrix (shape: (3,3))
    :param max_reprojection_err: maximum reprojection error
    :returns: views, tracks
    """
    optimizer, initial_values = sfm.get_optimizer(views, tracks, K)
    result = optimizer.optimize(initial_values)
    new_views, new_tracks = sfm.copy_results(views, tracks)
    assert(result.status == symforce.opt.optimizer.Optimizer.Status.SUCCESS)
    # Modifies views and tracks in-place
    sfm.store_results(new_views, new_tracks, K, result, max_reprojection_err=max_reprojection_err)
    return new_views, new_tracks


def show_reproj_results(views, tracks, K, distortion, print_raw_reproj=True, show_reproj_histogram=True):
    """ Show the results (errors) for reprojection

    :param views: views to use
    :param tracks: tracks to use
    :param K: camera matrix (shape: (3,3))
    :param distortion: distortion coefficients (array of length 4, or None)
    :param print_raw_reproj: if True, we print raw predictions using distortion model
    :param show_reproj_histogram: if True, we show histogram
    """
    
    # Get reprojection errors
    e_undistorted = [[] for view in views] # reprojection errors with respect to pre-processed image points
    e_raw = [[] for view in views] # reprojection errors for raw detections when distortion applied during projection
    for track in tracks:
        if not track['valid']:
            continue
        
        for match in track['matches']:
            view_id = match['view_id']
            feature_id = match['feature_id']
            view = views[view_id]
            e_undistorted[view_id].append(
                sfm.projection_error(K, view['R_inB_ofA'], view['p_inB_ofA'], track['p_inA'], view['pts'][feature_id]['pt2d'], distortion=None)
            )
            e_raw[view_id].append(
                sfm.projection_error(K, view['R_inB_ofA'], view['p_inB_ofA'], track['p_inA'], view['pts'][feature_id]['pt2d_raw'], distortion=distortion)
            )
    
    print('\nREPROJECTION ERRORS')

    # Text
    for i_view, (e_undist_i, e_raw_i, view) in enumerate(zip(e_undistorted, e_raw, views)):
        if len(e_undist_i) == 0:
            assert((view['R_inB_ofA'] is None) or (view['p_inB_ofA'] is None))
            continue

        assert(not ((view['R_inB_ofA'] is None) or (view['p_inB_ofA'] is None)))
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
    selected_views = [view for view in views if view['R_inB_ofA'] is not None]
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
                R, p = views[m['view_id']]['R_inB_ofA'], views[m['view_id']]['p_inB_ofA']
                proj = sfm.project(K, R, p, np.row_stack([t['p_inA']]), warn=False, distortion=distortion)[0]
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
