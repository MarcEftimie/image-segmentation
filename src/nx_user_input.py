import cv2

global clicked_points, image
clicked_points = []

RADIUS = 20
SCALE_FACTOR = 20

def point_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def on_click(event, x, y, flags, is_source):
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
    global clicked_points
    clicked_points = []
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_click, is_source)

    while(1):
        global image
        cv2.imshow('image', image)
        if cv2.waitKey(20) & 0xFF == 27:  # Break loop when 'ESC' is pressed
            break
    cv2.destroyAllWindows()

    return clicked_points

def get_user_input(inp_image):
    # Clone the image to not draw on the original image
    global image
    image = inp_image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # resize by scale factor of 10
    image = cv2.resize(image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    source_points = plant_seeds(is_source=True)
    sink_points = plant_seeds(is_source=False)
    return source_points, sink_points
