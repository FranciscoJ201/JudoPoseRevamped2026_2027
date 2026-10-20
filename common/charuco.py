import cv2


def get_aruco_dictionary(dict_name):
    """
    Retrieves the cv2.aruco predefined dictionary object matching the given name.

    Arguments:
        dict_name (str): Name string of the dictionary (e.g. 'DICT_5X5_50').

    Returns:
        cv2.aruco.Dictionary: Instantiated dictionary.
    """
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"[-] Unknown ArUco dictionary: {dict_name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))


def build_charuco_detector(squares_x, squares_y, square_len, marker_len, dict_name):
    """
    Builds the ChArUco board and its detector from physical board parameters.

    Arguments:
        squares_x (int): Number of squares along the X dimension.
        squares_y (int): Number of squares along the Y dimension.
        square_len (float): Square side length in meters.
        marker_len (float): ArUco marker side length in meters.
        dict_name (str): ArUco dictionary name (e.g. 'DICT_5X5_50').

    Returns:
        tuple: (board, detector) -> (cv2.aruco.CharucoBoard, cv2.aruco.CharucoDetector)
    """
    dictionary = get_aruco_dictionary(dict_name)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, dictionary)
    return board, cv2.aruco.CharucoDetector(board)
