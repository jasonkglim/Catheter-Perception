import cv2.aruco as aruco
import cv2
import numpy as np
import glob
import os
import pickle

def calibrate_intrinsics_charuco(glob_pattern, charuco_board, aruco_dict,
                                 min_markers=20):
    '''
    Calibrates camera intrinsics using a ChArUco board.
    '''
    all_corners, all_ids, img_size = [], [], None

    for fname in glob.glob(glob_pattern):
        img = cv2.imread(fname)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = img_gray.shape[::-1]

        detector = aruco.CharucoDetector(charuco_board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img_gray)

        all_corners.append(charuco_corners)
        all_ids.append(charuco_ids)

    # 3. Calibrate
    ret, K, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=charuco_board,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None
    )
    return ret, K, dist, rvecs, tvecs

def stereo_calibrate_charuco(glob1, glob2, charuco_board, aruco_dict,
                             K1, d1, K2, d2, flags=cv2.CALIB_FIX_INTRINSIC):
    objpoints, imgpts1, imgpts2 = [], [], []
    size = None

    # Pair up images
    for f1, f2 in zip(sorted(glob.glob(glob1)), sorted(glob.glob(glob2))):
        im1, im2 = cv2.imread(f1), cv2.imread(f2)
        g1, g2 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        if size is None: size = g1.shape[::-1]

        # Detect & interpolate
        c1, id1, _ = aruco.detectMarkers(g1, aruco_dict)
        c2, id2, _ = aruco.detectMarkers(g2, aruco_dict)
        r1, cc1, cid1 = aruco.interpolateCornersCharuco(c1, id1, g1, charuco_board)
        r2, cc2, cid2 = aruco.interpolateCornersCharuco(c2, id2, g2, charuco_board)
        if r1 < 20 or r2 < 20: continue

        # Build object points (same for both)
        objp = charuco_board.chessboardCorners[cid1.flatten()]
        objpoints.append(objp)
        imgpts1.append(cc1)
        imgpts2.append(cc2)

    # Stereo calibrate
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpts1, imgpts2,
        K1, d1, K2, d2, size,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER|cv2.TERM_CRITERIA_EPS,100,1e-5),
        flags=flags
    )
    print(f"Stereo RMS error: {rms:.4f}")
    return R, T


def camera_to_world_charuco(glob_pattern, charuco_board, aruco_dict, K, dist):
    for fname in glob.glob(glob_pattern):
        im = cv2.imread(fname); gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # 1. Detect markers + ChArUco
        markers, ids, _ = aruco.detectMarkers(gray, aruco_dict)
        ret, cc, cid = aruco.interpolateCornersCharuco(markers, ids, gray, charuco_board)
        if ret < 20: continue

        # 2. Get the corresponding 3D object points
        obj_pts = charuco_board.chessboardCorners[cid.flatten()]

        # 3. SolvePnP
        _, rvec, tvec = cv2.solvePnP(obj_pts, cc, K, dist)
        # 4. (Optional) compute reprojection error
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        err = np.linalg.norm(cc.reshape(-1,2) - proj.reshape(-1,2)) / len(proj)
        print(f"Cam→World PnP error: {err*1000:.2f} mm")
        return rvec, tvec

    raise RuntimeError("ChArUco board not found in any image")


if __name__ == "__main__":
    
    # Calculates and saves camera intrinsics and extrinisics for two cameras
    calibration_id = "05-08-25"
    calib_base_path = f"/home/arclab/catkin_ws/src/Catheter-Perception/camera_calibration/{calibration_id}"

    camera_calib_data = {
        "cam0": {
            "intrinsics": {
                "K": None,
                "d": None
            },
            "extrinsics": {
                "R": None,
                "T": None
            }
        },
        "cam1": {
            "intrinsics": {
                "K": None,
                "d": None
            },
            "extrinsics": {
                "R": None,
                "T": None
            }
        }
    }

    # Define ChArUco board used
    square_size = 0.006
    marker_length = 0.004
    board_size = (10, 10)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    charuco_board = aruco.CharucoBoard(
        size=board_size,
        squareLength=square_size, 
        markerLength=marker_length,
        dictionary=aruco_dict
    )

    # Calibrate instrinsics
    for i in range(2):
        intrinsic_calib_images_path = f"{calib_base_path}/charuco_calib_images/cam{i}/*.png"
        print(intrinsic_calib_images_path)
        rms_error, K, d, _, _ = calibrate_intrinsics_charuco(
            intrinsic_calib_images_path, charuco_board, aruco_dict
        )
        print(f"Charuco intrinsics for camera {i} RMS error: {rms_error:.4f}")
        camera_calib_data[f"cam{i}"]["intrinsics"]["K"] = K
        camera_calib_data[f"cam{i}"]["intrinsics"]["d"] = d

    # save calibration data
    with open(f"{calib_base_path}/camera_calib_data.pkl", "wb") as f:
        pickle.dump(camera_calib_data, f)
    
    print("Calibration complete.")
