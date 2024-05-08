# AE598 Project

### Dependencies
This project has dependencies that include all dependencies for the SfM homework in the course. That is, the standard dependencies plus rerun (as needed for the SfM homework).
In addition, PyKITTI and Pandas are also required dependencies for reproducing our results.
PyKITTI can be installed with:
``
pip install pykitti
``
Pandas can be installed through pip, or with:
``
mamba install pandas
``

Some functions in pose_metrics.py (which is not used for our final implementation and contains code that we did not end up using for our final results) expect that the "evo" package is installed (however, evo is used with subprocess calls--not imports--so there should be no import errors, even if evo is not installed, so long as the functions that call evo are not used).
To install evo:
``
pip install evo --upgrade --no-binary evo
``

### Data
The data for the project is hosted on Box, at the following link:
https://uofi.box.com/s/sdy9gdvqym1d7cz0g52tvad3jaly2q31

Note: this dataset file is quite large (~6 gb).

For proper reading of the data, please move the downloaded data from Box into a subdirectory called ``data``.
Specifically, the organization should be as follows (where the root ``/`` directory is the directory containing this README file):

- ``/root``
    - ``README.md``
    - ``data/``
        - ``kitti/``
            - ``2011_09_26/`` 
        - ``mav0/``
            - ``cam0``
            - ``cam1``
            - ``imu0``
            - ``leica0``
            - ``state_groundtruth_estimate0``
            - ``body.yaml``

### Code
The code is organized as follows:
``imu_preintegration.py``
    - Containscode for inertial measurement unit (IMU) pre-integration
``pose_metrics.py``
    - Contains code for pose metrics (mostly using the evo package). Ultimately, this code ended up not being used (it is legacy code from before the projected pivoted to focus more on IMU integration into two-view reconstruction rather than full-fledged visual-inertial odometry).
``sfm.py``
    - Implementation for structure-from-motion
``utils.py``
    - Miscellenous utilities for procedures like dataset loading and processing.
``vio.py``
    - Implementation of two-view reconstruction using IMU data (visual-inertial odometry)
``vo.py``
    - Implementation of two-view reconstruction without using IMU data (visual odometry)
``vio_benchmark.ipynb``
    - Jupyter Notebook for testing visual-inertial odometry.
``vo_benchmark.ipynb``
    - Jupyter Notebook for testing visual odometry.

The main code to interact with our implementation is in the Jupyter notebooks, which feature implementations using only visual odometry and using visual-inertial odometry.
More details can be found in the code itself.