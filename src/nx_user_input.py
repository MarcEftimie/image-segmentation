"""
Functions for getting user input for source and sink points.
"""

import cv2

global clicked_points, image
clicked_points = []

RADIUS = 20
SCALE_FACTOR = 20

def colorPixel(image, i, j, red=False):
    """
    Colors a pixel at a specified position in the global `image` array.

    Args:
        image (np.array): The image to draw on.
        i, j (int): The row and column indices of the pixel in the image.
        red (bool): Whether to color the pixel red or blue.

    Returns:
        None
    """
    try:
        if red:
            image[i][j] = (0, 0, 255)
        else:
            image[i][j] = (255, 0, 0)
    except:
        print(i, j)


def displayCut(image, cuts):
    """
    Displays the cut on the image by coloring the pixels on the cut.

    Args:
        image (np.array): The image on which the cut is to be displayed.
        cuts (list of tuples): The list of cuts where each cut is a tuple of two pixel indices.

    Returns:
        np.array: The image with the cut displayed.
    """
    r, c, _ = image.shape
    for c in cuts:
        if (
            c[0] != image.size - 2
            and c[0] != image.size - 1
            and c[1] != image.size - 2
            and c[1] != image.size - 1
        ):
            colorPixel(image, c[0] // r, c[0] % r)
            colorPixel(image, c[1] // r, c[1] % r)
    return image


def point_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        p1 (tuple): The first point, a tuple of (x, y).
        p2 (tuple): The second point, a tuple of (x, y).

    Returns:
        (float): The Euclidean distance between the two points.
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def on_click(event, x, y, flags, is_source):
    """
    Handle mouse click events to draw circles and record clicked points.

    Args:
        event (int): The event type.
        x (int): The x-coordinate of the mouse click.
        y (int): The y-coordinate of the mouse click.
        flags (int): Any relevant flags.
        is_source (bool): Whether the click is on the source image.
    """
    global clicked_points, image
    if event == cv2.EVENT_LBUTTONDOWN:
        if is_source:
            cv2.circle(image, (x, y), 20, (0, 255, 0), -1)
        else:
            cv2.circle(image, (x, y), 20, (255, 0, 0), -1)
        for i in range(x - RADIUS, x + RADIUS):
            for j in range(y - RADIUS, y + RADIUS):
                if point_distance((i, j), (x, y)) <= RADIUS:
                    clicked_points.append((i//SCALE_FACTOR, j//SCALE_FACTOR))


def plant_seeds(is_source):
    """
    Initialize the image window and set the mouse callback function.

    Args:
        is_source (bool): Whether the seeds are being planted on the source image.

    Returns:
        (list): The list of clicked points.
    """
    global clicked_points, image
    clicked_points = []
    if is_source:
        cv2.namedWindow('Click on the source points and press ESC')
        cv2.setMouseCallback('Click on the source points and press ESC', on_click, is_source)
    else:
        cv2.namedWindow('Click on the sink points and press ESC')
        cv2.setMouseCallback('Click on the sink points and press ESC', on_click, is_source)

    while(1):
        if is_source:
            cv2.imshow('Click on the source points and press ESC', image)
        else:
            cv2.imshow('Click on the sink points and press ESC', image)
        if cv2.waitKey(20) & 0xFF == 27:  # Break loop when 'ESC' is pressed
            break
    cv2.destroyAllWindows()

    return clicked_points


def get_user_input(inp_image):
    """
    Get user input for source and sink points on the image.

    Args:
        inp_image (ndarray): The input image.

    Returns:
        (tuple): The source and sink points.
    """
    # Clone the image to not draw on the original image
    global image
    image = inp_image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # resize by scale factor of 10
    image = cv2.resize(image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    source_points = plant_seeds(is_source=True)
    sink_points = plant_seeds(is_source=False)
    return source_points, sink_points
