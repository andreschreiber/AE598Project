import cv2
import yaml
import secrets
import numpy as np
from pathlib import Path


def read_yaml(file):
    """ Reads a YAML file """
    with open(file) as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            return None


def create_rng(seed=None):
    """ Creates a numpy random number generator """
    if seed is None:
        seed = secrets.randbits(32)
    print(f'seeding RNG with {seed}')
    rng = np.random.default_rng(seed)
    return rng


def read_image_mav(file):
    """ Reads a file from the MAV dataset """
    return cv2.imread(str(file))[:,:,0]


def collect_visual_inertial_data(imu_csv, cam_csv, gt_csv, image_folder_path, filter_by_gt=True):
    """ Collects visual-inertial data from CSVs

    :param imu_csv: imu data csv
    :param cam_csv: camera data csv
    :param gt_csv: csv for groundtruth estimates
    :param image_folder_path: path for image folder
    :param filter_by_gt: if True, we remove the start and end of the data
                         to ensure only samples within very small groundtruth deltas
                         are included.
    :return: list of dictionaries of form:
        {
            'time': time in seconds
            'w_x': angular velocity in x
            'w_y': angular velocity in y
            'w_z': angular velocity in z
            'a_x': angular velocity in x
            'a_y': angular velocity in y
            'a_z': angular velocity in z
            'image_file': image file or None if no image data for this timestep
            'groundtruth': {
                'delta': absolute delta time to the nearest groundtruth sample
                'p_x': position x,
                'p_y': position y,
                'p_z': position z,
                'q_w': rotation quaternion w
                'q_x': rotation quaternion x
                'q_y': rotation quaternion y
                'q_z': rotation quaternion z,
                'v_x': velocity x
                'v_y': velocity y
                'v_z': velocity z
                'b_w_x': IMU angular velocity bias (x)
                'b_w_y': IMU angular velocity bias (y)
                'b_w_z': IMU angular velocity bias (z)
                'b_a_x': IMU acceleration bias (x)
                'b_a_y': IMU acceleration bias (y)
                'b_a_z': IMU acceleration bias (z)
            }
        } 
    """

    samples = []
    gt_csv_times = gt_csv['#timestamp']

    for i in range(len(imu_csv)):
        imu_entry = imu_csv.iloc[i]
        ts = imu_entry['#timestamp [ns]']

        sample = {}

        # Get IMU data
        sample['time'] = ts * (1e-9) # convert from ns to s

        sample['w_x'] = imu_entry['w_RS_S_x [rad s^-1]']
        sample['w_y'] = imu_entry['w_RS_S_y [rad s^-1]']
        sample['w_z'] = imu_entry['w_RS_S_z [rad s^-1]']

        sample['a_x'] = imu_entry['a_RS_S_x [m s^-2]']
        sample['a_y'] = imu_entry['a_RS_S_y [m s^-2]']
        sample['a_z'] = imu_entry['a_RS_S_z [m s^-2]']

        # Get camera data
        cam_entry_loc = (cam_csv['#timestamp [ns]'] == ts)
        assert(cam_entry_loc.sum() in [0,1]) # make sure we don't have two camera entries with same time (shouldn't happen)
        if cam_entry_loc.sum() == 1:
            sample['image_file'] = Path(image_folder_path) / Path(cam_csv[cam_entry_loc]['filename'].iloc[0])
        else:
            sample['image_file'] = None

        # Get groundtruth state estimate data
        closest_gt_idx = np.abs(gt_csv_times - ts).argmin()
        gt_entry = gt_csv.iloc[closest_gt_idx]
        sample['groundtruth'] = {
            'delta': abs(gt_entry['#timestamp'] - ts) * (1e-9), # make sure to convert to seconds!
            'p_x': gt_entry[' p_RS_R_x [m]'],
            'p_y': gt_entry[' p_RS_R_y [m]'],
            'p_z': gt_entry[' p_RS_R_z [m]'],
            'q_w': gt_entry[' q_RS_w []'],
            'q_x': gt_entry[' q_RS_x []'],
            'q_y': gt_entry[' q_RS_y []'],
            'q_z': gt_entry[' q_RS_z []'],
            'v_x': gt_entry[' v_RS_R_x [m s^-1]'],
            'v_y': gt_entry[' v_RS_R_y [m s^-1]'],
            'v_z': gt_entry[' v_RS_R_z [m s^-1]'],
            'b_w_x': gt_entry[' b_w_RS_S_x [rad s^-1]'],
            'b_w_y': gt_entry[' b_w_RS_S_y [rad s^-1]'],
            'b_w_z': gt_entry[' b_w_RS_S_z [rad s^-1]'],
            'b_a_x': gt_entry[' b_a_RS_S_x [m s^-2]'],
            'b_a_y': gt_entry[' b_a_RS_S_y [m s^-2]'],
            'b_a_z': gt_entry[' b_a_RS_S_z [m s^-2]'],
        }

        samples.append(sample)
    
    # Make sure the samples are time sorted.
    samples = sorted(samples, key=lambda x: x['time'])
    if filter_by_gt:
        # Get points where delta is very small
        matches = (np.abs(np.array([s['groundtruth']['delta'] for s in samples]) - 0.0) < 1e-9)
        first_idx = np.nonzero(matches)[0][0]
        last_idx = np.nonzero(matches)[0][-1]
        samples = samples[first_idx:last_idx]

    return samples


def get_index_of_next_image(samples, current_index):
    """ Returns index of next sample that has a camera image """
    for i in range(current_index, len(samples)):
        if samples[i]['image_file'] is not None:
            return i
    raise ValueError("No more frames")