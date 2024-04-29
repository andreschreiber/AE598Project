import cv2
import yaml
import secrets
import numpy as np
import pandas as pd
import pykitti
from pathlib import Path
from scipy.spatial.transform import Rotation

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


def read_image(file):
    """ Reads an image as grayscale (MAV or KITTI) """
    return cv2.imread(str(file))[:,:,0]


def read_data_mav(mav_video_folder):
    """ Reads the EuRoC MAV data at the provided folder
    
    :param mav_video_folder: folder for mav data
    :returns: dictionary of resulting data
    """
    # Read the camera data
    cam0_sensor = read_yaml(mav_video_folder / 'cam0/sensor.yaml')

    cam0_K = np.array([
        [cam0_sensor['intrinsics'][0], 0, cam0_sensor['intrinsics'][2]],
        [0, cam0_sensor['intrinsics'][1], cam0_sensor['intrinsics'][3]],
        [0, 0, 1]
    ])
    cam0_distortion = np.array(cam0_sensor['distortion_coefficients'])
    cam0_resolution = np.array(cam0_sensor['resolution']) # W,H
    cam0_extrinsics = np.array(cam0_sensor['T_BS']['data']).reshape(4,4)

    # Read the IMU data
    imu_sensor = read_yaml(mav_video_folder / 'imu0/sensor.yaml')

    imu_extrinsics = np.array(imu_sensor['T_BS']['data']).reshape(4,4)
    imu_gyroscope_noise_density = imu_sensor['gyroscope_noise_density']
    imu_gyroscope_random_walk = imu_sensor['gyroscope_random_walk']
    imu_accelerometer_noise_density = imu_sensor['accelerometer_noise_density']
    imu_accelerometer_random_walk = imu_sensor['accelerometer_random_walk']

    # Read the frame data
    cam0_csv = pd.read_csv(mav_video_folder / 'cam0/data.csv')
    imu0_csv = pd.read_csv(mav_video_folder / 'imu0/data.csv')
    states_csv = pd.read_csv(mav_video_folder / 'state_groundtruth_estimate0/data.csv')
    visual_inertial_data = collect_visual_inertial_data_mav(imu0_csv, cam0_csv, states_csv, mav_video_folder / Path('cam0/data'))

    return {
        'cam0_K': cam0_K,
        'cam0_distortion': cam0_distortion,
        'cam0_extrinsics': cam0_extrinsics,
        'imu_extrinsics': imu_extrinsics,
        'imu_gyroscope_noise_density': imu_gyroscope_noise_density,
        'imu_gyroscope_random_walk': imu_gyroscope_random_walk,
        'imu_accelerometer_noise_density': imu_accelerometer_noise_density,
        'imu_accelerometer_random_walk': imu_accelerometer_random_walk,
        'visual_inertial_data': visual_inertial_data
    }


def collect_visual_inertial_data_mav(imu_csv, cam_csv, gt_csv, image_folder_path, filter_by_gt=True):
    """ Collects visual-inertial data from EuRoC CSVs

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
            'a_x': acceleration in x
            'a_y': acceleration in y
            'a_z': acceleration in z
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


def euler_to_quat(roll, pitch, yaw):
    """ Converts roll, pitch, and yaw to a quaternion (x,y,z,w) """
    # Convert from euler angles to SO3
    # https://pypose.org/docs/main/generated/pypose.euler2SO3/?highlight=so3#pypose.euler2SO3
    sin_a, cos_a = np.sin(roll*0.5), np.cos(roll*0.5)
    sin_b, cos_b = np.sin(pitch*0.5), np.cos(pitch*0.5)
    sin_y, cos_y = np.sin(yaw*0.5), np.cos(yaw*0.5)

    return np.array([
        sin_a * cos_b * cos_y - cos_a * sin_b * sin_y,
        cos_a * sin_b * cos_y + sin_a * cos_b * sin_y,
        cos_a * cos_b * sin_y - sin_a * sin_b * cos_y,
        cos_a * cos_b * cos_y + sin_a * sin_b * sin_y
    ])

def read_data_kitti(base_path, date, drive):
    """ Read data from KITTI 
    
    :param base_path: base KITTI path
    :param date: date of drive
    :param drive: drive to use
    :returns: dictionary of resulting data
    """
    
    pyk_raw = pykitti.raw(base_path, date, drive)

    cam0_K = pyk_raw.calib.K_cam0
    cam0_distortion = None # camera images already undistorted
    cam0_extrinsics = pyk_raw.calib.T_cam0_imu # this is from IMU to camera frame
    
    imu_extrinics = np.eye(4)

    visual_inertial_data = collect_visual_inertial_data_kitti(pyk_raw)

    return {
        'cam0_K': cam0_K,
        'cam0_distortion': cam0_distortion,
        'cam0_extrinsics': cam0_extrinsics,
        'imu_extrinics': imu_extrinics,
        'visual_inertial_data': visual_inertial_data
    }


def collect_visual_inertial_data_kitti(pyk_raw):
    """ Collects visual-inertial data from KITTI

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
            'a_x': acceleration in x
            'a_y': acceleration in y
            'a_z': acceleration in z
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
            }
        } 
    """

    seq_len = len(pyk_raw.timestamps)
    assert(seq_len == len(pyk_raw.oxts))
    assert(seq_len == len(pyk_raw.cam0_files))

    samples = []

    for i in range(seq_len):
        sample = {}
        sample['time'] = (pyk_raw.timestamps[i] - pyk_raw.timestamps[0]).total_seconds()
        packet = pyk_raw.oxts[i].packet

        sample['w_x'] = packet.wx
        sample['w_y'] = packet.wy
        sample['w_z'] = packet.wz

        sample['a_x'] = packet.ax
        sample['a_y'] = packet.ay
        sample['a_z'] = packet.az

        sample['image_file'] = Path(pyk_raw.cam0_files[i])

        # rotation
        rot_quat = euler_to_quat(packet.roll, packet.pitch, packet.yaw)
        rot_mat = Rotation.from_quat(rot_quat).as_matrix()

        # velocity
        vel = rot_mat @ np.array([
            packet.vf,
            packet.vl,
            packet.vu
        ])

        # we could probably also extract rotation from this instead
        pos = pyk_raw.oxts[i].T_w_imu[0:3,3]

        sample['groundtruth'] = {
            'delta': 0.0,
            'p_x': pos[0],
            'p_y': pos[1],
            'p_z': pos[2],
            'q_w': rot_quat[3],
            'q_x': rot_quat[0],
            'q_y': rot_quat[1],
            'q_z': rot_quat[2],
            'v_x': vel[0],
            'v_y': vel[1],
            'v_z': vel[2]
        }

        samples.append(sample)
    
    return samples


def get_index_of_next_image(samples, current_index):
    """ Returns index of next sample that has a camera image """
    for i in range(current_index, len(samples)):
        if samples[i]['image_file'] is not None:
            return i
    raise ValueError("No more frames")