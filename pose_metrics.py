# Author: Andre Schreiber
# This file contains code to help with using the evo VO/VIO evaluation package.
# Note: this expects that evo is installed: pip install evo --upgrade --no-binary evo

import json
import shutil
import zipfile
import subprocess
import numpy as np
import cv2

from pathlib import Path
from scipy.spatial.transform import Rotation


def create_tum_pose_file(output_file, times, positions, quaternions):
    """ Create a pose txt file in the tum format for evo
    
    :param output_file: file to use as output (will be deleted and overwritten if exists)
    :param times: list of times
    :param positions: list of positions (list of 3-vectors)
    :param quaternions: list of quaternions (list of quaternions of form (x,y,z,w))
    :return: True if successful creation of file
    """
    output_file = Path(output_file)
    if output_file.is_file():
        output_file.unlink()

    assert(len(times) == len(positions) and len(positions) == len(quaternions))
    file_lines = []
    for t, p, q in zip(times, positions, quaternions):
        # expect quaternions to be in form (x,y,z,w)
        file_lines.append(
            "{} {} {} {} {} {} {} {}\n".format(t, p[0], p[1], p[2], q[0], q[1], q[2], q[3])
        )
    with open(output_file, 'w') as f:
        f.writelines(file_lines)
    
    return output_file.is_file()


def create_tum_data_from_views(times, views):
    """ Helper utility function to create TUM data from views 
    
    :param times: times for each view
    :param views: list of views
    :return: (times, positions, quaternions)
    """
    assert(len(times) == len(views))

    positions = []
    quaternions = []
    for view in views:
        assert(view['R_inB_ofA'] is not None and view['p_inB_ofA'] is not None)
        positions.append(view['p_inB_ofA'])
        quaternions.append(Rotation.from_matrix(view['R_inB_ofA']).as_quat())

    return (times, positions, quaternions)


def create_tum_data_from_groundtruth(visual_inertial_data):
    """ Helper function to create TUM data from visual inertial data read in via utils.py
    
    :param visual_inertial_data: visual inertial data as read by collect_visual_inertial_data in utils.py
    :return: (times, positions, quaternions)
    """

    times, positions, quaternions = [], [], []
    for data in visual_inertial_data:
        times.append(data['time'])
        positions.append(np.array([
            data['groundtruth']['p_x'], data['groundtruth']['p_y'], data['groundtruth']['p_z']
        ]))
        quaternions.append(np.array([
            data['groundtruth']['q_x'], data['groundtruth']['q_y'], data['groundtruth']['q_z'],  data['groundtruth']['q_w']
        ]))
    
    return (times, positions, quaternions)


def compute_ape(traj_1, traj_2,
                traj_1_file, traj_2_file,
                results_file, results_dir, align='none',
                verbose=True, print_results=True, cleanup=True):
    """ Execute APE calculation using the evo package
    
    :param traj_1: first trajectory - tuple of form (times, positions, quaternions)
    :parma traj_2: second trajectory - tuple of form (times, positions, quaternions)
    :param traj_1_file: filename for traj_1 temporary file
    :param traj_2_file: filename for traj_2 temporary file
    :param results_file: results file to save to (zip file)
    :param results_dir: directory to extract results zip to
    :param align: method of alignment (none, scale, pose, or posescale)
    :param verbose: if True, the output from evo will be printed
    :param print_results: if True, we print the results returned by evo
    :param cleanup: if True, we delete all temporary files
    :return: result metrics as a dictionary
    """

    traj_1_file = Path(traj_1_file)
    traj_2_file = Path(traj_2_file)

    results_file = Path(results_file)
    results_dir = Path(results_dir)

    # Remove the results files if they exist
    if results_file.is_file():
        results_file.unlink()
    if results_dir.exists():
        shutil.rmtree(results_dir)

    # Create first trajectory file
    if create_tum_pose_file(traj_1_file, *traj_1):
        if verbose:
            print("Trajectory 1 temporary file created")
    else:
        raise RuntimeError("Failed to create trajectory 1 temporary file")

    # Create second trajectory file
    if create_tum_pose_file(traj_2_file, *traj_2):
        if verbose:
            print("Trajectory 2 temporary file created")
    else:
        raise RuntimeError("Failed to create trajectory 2 temporary file")

    # Prepare the command for calling evo
    call_array = ['evo_ape', 'tum', str(traj_1_file), str(traj_2_file)]
    if align == 'none':
        pass
    elif align == 'scale':
        call_array.append("-s")
    elif align == 'pose':
        call_array.append("-a")
    elif align == 'posescale':
        call_array.append("-as")
    else:
        raise ValueError("Invalid alignment")
    
    call_array.append("--save_results")
    call_array.append(str(results_file))

    if verbose:
        subprocess.call(call_array)
    else: # We can redirect output to prevent verbose printing
        subprocess.call(call_array, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    # evo saves results to zip, so we unzip and then find the stats.json file with the results.
    evaluation_results = None
    if results_file.exists():
        with zipfile.ZipFile(results_file, 'r') as zipped:
            zipped.extractall(results_dir)
        if results_dir.exists():
            stats_file = results_dir / Path('stats.json')
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    evaluation_results = json.load(f)
            else:
                raise FileNotFoundError("Failed to find stats.json file")
        else:
            raise FileNotFoundError("APE evaluator result extraction failed")
    else:
        raise FileNotFoundError("APE evaluator failed")

    # Print results if requested
    if print_results:
        print("\nAPE EVALUATION (alignment = {})".format(align))
        for k in evaluation_results.keys():
            print("{} = {:.3f}".format(k, evaluation_results[k]))

    # If cleanup is True, we clean up the temporary files
    if cleanup:
        if traj_1_file.exists():
            traj_1_file.unlink()
        if traj_2_file.exists():
            traj_2_file.unlink()
        
        if results_file.exists():
            results_file.unlink()
        if results_dir.exists():
            shutil.rmtree(results_dir)

    # Return the results
    return evaluation_results


def rotational_error(R1, R2):
    """ Compute a scalar metric for error between two rotations
    
    Source: https://stackoverflow.com/questions/6522108/error-between-two-rotations

    :param R1: first rotation matrix (shape: (3,3))
    :param R2: second rotation matrix (shape: (3,3))
    """
    rotator, _ = cv2.Rodrigues(R1.dot(R2.T))
    error = np.linalg.norm(rotator)
    return error


def zero_transform_tum_data(times, positions, quaternions, scaling=1.0):
    """ Zero transforms TUM data to make the first entry have 0 position and identity rotation
    
    :param times: list of times
    :param positions: list of positions (list of 3-vectors)
    :param quaternions: list of quaternions (list of 4-vectors representing quaternions x,y,z,w)
    :param scaling: scaling factor
    :returns: [times, new_positions, new_quaternions]
    """
    pose_offset = positions[0] # want to move to zero
    rotation_offset = Rotation.from_quat(quaternions[0]).as_matrix().T # get inverse of rotation
    new_positions, new_quaternions = [], []
    for (p, q) in zip(positions, quaternions):
        new_positions.append((p - pose_offset) * scaling)
        new_quaternions.append(
             #TODO: is order right?
            Rotation.from_matrix((rotation_offset @ Rotation.from_quat(q).as_matrix())).as_quat()
        )
    return times, new_positions, new_quaternions


def compute_pose_error(traj_1, traj_2, align='posescale', print_results=True):
    """ Compute step-wise pose error in rotation and translation
    
    :param traj_1: first trajectory - tuple of form (times, positions, quaternions)
    :parma traj_2: second trajectory - tuple of form (times, positions, quaternions)
    :param align: method of alignment (none, pose, or posescale)
        - if scale is used, then the first trajectory should be the "ground truth"
    :param print_results: if True, we print the results
    :return: result metrics as a dictionary
    """

    assert(align in ['pose', 'posescale', 'none'])

    # Do alignment
    if align == 'pose' or align == 'posescale':
        traj_2_scaling = np.linalg.norm(traj_1[1][-1] - traj_1[1][0], ord=2) if align == 'posescale' else 1.0
        traj_1 = zero_transform_tum_data(*traj_1, scaling=1.0)
        traj_2 = zero_transform_tum_data(*traj_2, scaling=traj_2_scaling)

    errors = {
        'translation': [],
        'rotation': []
    }
    for i in range(len(traj_1[0])):
        pose1, pose2 = traj_1[1][i], traj_2[1][i]
        quat1, quat2 = traj_1[2][i], traj_2[2][i]
        rot1, rot2 = Rotation.from_quat(quat1).as_matrix(), Rotation.from_quat(quat2).as_matrix()
        errors['translation'].append(np.linalg.norm(pose1 - pose2, ord=2))
        errors['rotation'].append(rotational_error(rot1, rot2))
    errors['translation'] = np.array(errors['translation'])
    errors['rotation'] = np.array(errors['rotation'])

    if print_results:
        print("Avg error in translation = {:.3f} (min = {:.3f}, max = {:.3f}, final = {:.3f})".format(
            errors['translation'].mean(), errors['translation'].min(), errors['translation'].max(), errors['translation'][-1]
        ))
        print("Avg error in rotation = {:.3f} (min = {:.3f}, max = {:.3f}, final = {:.3f})".format(
            errors['rotation'].mean(), errors['rotation'].min(), errors['rotation'].max(), errors['rotation'][-1]
        ))

    # Return the results
    return errors
