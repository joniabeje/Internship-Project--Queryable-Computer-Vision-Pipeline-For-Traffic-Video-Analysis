import cv2
from util import get_limits, refine_mask
from ultralytics import YOLO
import cvzone
import math
from db_connector import create_connection, drop_tables, create_tables, insert_car_count, insert_or_update_frame, ensure_all_frames_recorded

def main():
    colors = {
        'Red': [0, 0, 255],  # red in BGR
        'Yellow': [0, 255, 255],  # yellow in BGR
        'Green': [0, 255, 0]  # green in BGR
    }

    connection = create_connection("127.0.0.1", "root", "12jo34ni", "traffic_db1")
    if connection is not None:
        drop_tables(connection)
        create_tables(connection)
    else:
        print("Failed to create a database connection.")
        return

    cap = cv2.VideoCapture('/Users/joni/Downloads/Mass_10th (1).MP4')

    if not cap.isOpened():
        print("Error: Failed to load video.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 10, (frame_width, frame_height))

    roi_traffic_light = select_roi(frame, "Select ROI for Traffic Light")
    if roi_traffic_light is None:
        print("No ROI selected for Traffic Light.")
        return

    rois_crosswalks = []
    for i in range(4):
        roi_crosswalk = select_roi(frame, f"Select ROI for Crosswalk {i + 1}")
        if roi_crosswalk is None:
            print(f"No ROI selected for Crosswalk {i + 1}.")
            return
        rois_crosswalks.append(roi_crosswalk)

    pixels_per_meter = calibrate_pixels_per_meter(rois_crosswalks)
    print(f"Calibrated pixels per meter: {pixels_per_meter}")

    model = YOLO("yolov8s-world.pt")
    classNames = model.names

    frame_id = 0
    next_custom_id = 1
    yolo_to_custom_id = {}
    detection_history = {}  # Track the detection history for each ID

    # Counters and tracking dictionaries
    vehicle_count = 0
    people_count = 0
    vehicle_ids = {}
    people_ids = {}
    MIN_DETECTION_FRAMES = 2  # Minimum number of frames an object must be detected to be counted

    # Counters for specific vehicle types
    car_count = 0
    truck_count = 0
    bus_count = 0
    bicycle_count = 0

    # Global variables for speed calculation
    vehicle_positions = {}
    vehicle_speeds = {}
    speed_histories = {}
    fps = 10  # Frames per second
    pixels_per_meter = 10  # This needs to be calibrated for your specific video

    max_frame_id = 0  # Initialize max_frame_id

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Captured all frames")
            break

        frame_id += 1
        max_frame_id = max(max_frame_id, frame_id)  # Update max_frame_id

        current_time = calculate_time_from_frame(frame_id, start_time="06:00:27:0", fps=10)
        print(f"Frame {frame_id}: {current_time}")

        traffic_light_color = process_traffic_light(frame, roi_traffic_light, colors)

        # Update the all_frames table for every frame
        insert_or_update_frame(connection, frame_id, current_time, traffic_light_color)

        results = model.track(frame, persist=True)  # Use tracking instead of detection

        persons_detected = []
        bicycles_detected = []
        vehicles_detected = []

        if results:
            for r in results:
                boxes = r.boxes
                if boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()  # Use YOLOv8's tracker ID
                else:
                    track_ids = [None] * len(boxes)  # Create a list of None if no IDs are present

                for box, track_id in zip(boxes, track_ids):
                    if track_id is None:
                        continue
                    if track_id not in yolo_to_custom_id:
                        yolo_to_custom_id[track_id] = next_custom_id
                        next_custom_id += 1

                    custom_id = yolo_to_custom_id[track_id]

                    # Track the detection history for each ID
                    if custom_id not in detection_history:
                        detection_history[custom_id] = 0
                    detection_history[custom_id] += 1

                    cls = int(box.cls[0])

                    if classNames[cls].lower() == "traffic light":
                        continue  # Skip processing for traffic lights

                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    jaywalking = "N/A"
                    red_light_dashing = "No"
                    speed = 0

                    if classNames[cls].lower() == "person":
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        if is_inside_roi(center_x, center_y, rois_crosswalks[0]) or is_inside_roi(center_x, center_y,
                                                                                                  rois_crosswalks[2]):
                            jaywalking = "Yes" if traffic_light_color == "Green" else "No"
                        elif is_inside_roi(center_x, center_y, rois_crosswalks[1]) or is_inside_roi(center_x, center_y,
                                                                                                    rois_crosswalks[3]):
                            jaywalking = "Yes" if traffic_light_color == "Red" else "No"
                        persons_detected.append((x1, y1, x2, y2, conf, custom_id, jaywalking))
                        if custom_id not in people_ids:
                            people_ids[custom_id] = 0
                        people_ids[custom_id] += 1
                        if people_ids[custom_id] == MIN_DETECTION_FRAMES:
                            people_count += 1
                    elif classNames[cls].lower() == "bicycle":
                        bicycles_detected.append((x1, y1, x2, y2, conf, custom_id))
                        if custom_id not in vehicle_ids:
                            vehicle_ids[custom_id] = 0
                        vehicle_ids[custom_id] += 1
                        if vehicle_ids[custom_id] == MIN_DETECTION_FRAMES:
                            bicycle_count += 1
                    else:  # Treat all other objects as vehicles
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        if custom_id in vehicle_positions:
                            prev_pos = vehicle_positions[custom_id]
                            speed = calculate_speed(prev_pos, (center_x, center_y), fps, pixels_per_meter)
                            speed = round(speed)  # Round speed to nearest integer
                        else:
                            speed = 0
                        vehicle_positions[custom_id] = (center_x, center_y)

                        if custom_id not in speed_histories:
                            speed_histories[custom_id] = []
                        average_speed = update_speed_average(speed_histories[custom_id], speed)

                        vehicle_speeds[custom_id] = average_speed  # Use average_speed instead of speed

                        if classNames[cls].lower() in ["car", "truck", "bus"] and custom_id in red_light_dashers:
                            red_light_dashing = "Yes"

                        vehicles_detected.append((x1, y1, x2, y2, conf, cls, custom_id, red_light_dashing, speed))

                        if custom_id not in vehicle_ids:
                            vehicle_ids[custom_id] = 0
                        vehicle_ids[custom_id] += 1
                        if vehicle_ids[custom_id] == MIN_DETECTION_FRAMES:
                            vehicle_count += 1

                            # Increment specific vehicle type counts
                            if classNames[cls].lower() == "car":
                                car_count += 1
                            elif classNames[cls].lower() == "truck":
                                truck_count += 1
                            elif classNames[cls].lower() == "bus":
                                bus_count += 1

                    if detection_history[custom_id] > 2:  # Ignore detections less than 2 frames
                        insert_car_count(connection, frame_id, custom_id, classNames[cls], conf,
                                         traffic_light_color, jaywalking, red_light_dashing, speed, current_time)

        check_jaywalking(frame, rois_crosswalks, persons_detected, traffic_light_color)
        check_red_light_dasher(frame, [(v[0], v[1], v[2], v[3], v[4], v[5], v[6]) for v in vehicles_detected],
                               traffic_light_color, rois_crosswalks, classNames)
        check_speeding(frame, vehicles_detected)

        # Draw bounding boxes around detected bicycles
        for bicycle in bicycles_detected:
            x1, y1, x2, y2, conf, track_id = bicycle
            cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=5, rt=1)
            cvzone.putTextRect(frame, f'bicycle {conf}', (max(0, x1), max(0, y1)), scale=1, thickness=2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add text for vehicle counts
        cv2.putText(frame, f'Cars: {car_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Trucks: {truck_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Buses: {bus_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Bicycles: {bicycle_count}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Add text for people count
        cv2.putText(frame, f'People: {people_count}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Write the frame to the video file
        out.write(frame)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # After processing all frames, ensure all frames are recorded
    ensure_all_frames_recorded(connection, max_frame_id)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    connection.close()

def calculate_speed(prev_pos, current_pos, fps, pixels_per_meter):
    dx = current_pos[0] - prev_pos[0]
    dy = current_pos[1] - prev_pos[1]
    distance_pixels = math.sqrt(dx**2 + dy**2)
    distance_meters = distance_pixels / pixels_per_meter
    time_seconds = 1 / fps
    speed = distance_meters / time_seconds
    return speed

def calibrate_pixels_per_meter(roi_crosswalks):
    known_distance_meters = 5  # Adjust this to a known distance in your video
    _, _, w, _ = roi_crosswalks[0]  # Assuming the width of the first crosswalk is the known distance
    pixels_per_meter = w / known_distance_meters
    return pixels_per_meter

def update_speed_average(speed_history, new_speed, window_size=5):
    speed_history.append(new_speed)
    if len(speed_history) > window_size:
        speed_history.pop(0)
    return sum(speed_history) / len(speed_history)

def select_roi(frame, window_name):
    roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    if roi[2] > 0 and roi[3] > 0:
        return roi
    else:
        return None

def process_traffic_light(frame, roi, colors):
    x, y, w, h = roi
    roi_frame = frame[y:y + h, x:x + w]
    hsvImage = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    detected_color = None

    limits_red = get_limits('Red')
    mask1_red = cv2.inRange(hsvImage, limits_red[0][0], limits_red[0][1])
    mask2_red = cv2.inRange(hsvImage, limits_red[1][0], limits_red[1][1])
    mask_red = cv2.bitwise_or(mask1_red, mask2_red)
    refined_mask_red = refine_mask(mask_red)
    contours_red, _ = cv2.findContours(refined_mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red = len(contours_red) > 0

    limits_yellow = get_limits('Yellow')
    mask_yellow = cv2.inRange(hsvImage, limits_yellow[0], limits_yellow[1])
    refined_mask_yellow = refine_mask(mask_yellow)
    contours_yellow, _ = cv2.findContours(refined_mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow = len(contours_yellow) > 0

    limits_green = get_limits('Green')
    mask_green = cv2.inRange(hsvImage, limits_green[0], limits_green[1])
    refined_mask_green = refine_mask(mask_green)
    contours_green, _ = cv2.findContours(refined_mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green = len(contours_green) > 0

    ltx, lty = x, y - 10

    if red and not yellow:
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors['Red'], 2)
        cv2.putText(frame, 'Red', (ltx, lty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, colors['Red'], 2, cv2.LINE_AA)
        detected_color = 'Red'
    elif yellow:
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors['Yellow'], 2)
        cv2.putText(frame, 'Yellow', (ltx, lty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, colors['Yellow'], 2, cv2.LINE_AA)
        detected_color = 'Yellow'
    elif green:
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors['Green'], 2)
        cv2.putText(frame, 'Green', (ltx, lty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, colors['Green'], 2, cv2.LINE_AA)
        detected_color = 'Green'

    return detected_color

def get_center_point(x1, y1, x2, y2):
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y

def is_inside_roi(x, y, roi):
    rx, ry, rw, rh = roi
    return rx < x < rx + rw and ry < y < ry + rh

def check_jaywalking(frame, rois, persons_detected, traffic_light_color):
    for person in persons_detected:
        if len(person) == 6:
            px1, py1, px2, py2, conf, track_id = person
        elif len(person) == 7:
            px1, py1, px2, py2, conf, track_id, _ = person  # Ignore pre-calculated jaywalking status
        else:
            continue  # Skip if the person tuple doesn't have the expected number of elements

        center_x, center_y = (px1 + px2) // 2, (py1 + py2) // 2
        jaywalking_status = None  # Default status is None

        # Check for bottom (index 0) and top (index 2) crosswalks
        if is_inside_roi(center_x, center_y, rois[0]) or is_inside_roi(center_x, center_y, rois[2]):
            if traffic_light_color == "Red":
                jaywalking_status = "Not Jaywalking"
            elif traffic_light_color == "Green":
                jaywalking_status = "Jaywalking"

        # Check for left (index 1) and right (index 3) crosswalks
        elif is_inside_roi(center_x, center_y, rois[1]) or is_inside_roi(center_x, center_y, rois[3]):
            if traffic_light_color == "Green":
                jaywalking_status = "Not Jaywalking"
            elif traffic_light_color == "Red":
                jaywalking_status = "Jaywalking"

        # Draw bounding boxes and labels based on jaywalking status
        if jaywalking_status == "Jaywalking":
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)  # Red box for jaywalker
            cv2.putText(frame, f'Jaywalker {conf}', (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f'ID: {track_id}', (px1, py1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif jaywalking_status == "Not Jaywalking":
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)  # Green box for non-jaywalker
            cv2.putText(frame, f'Not Jaywalking {conf}', (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f'ID: {track_id}', (px1, py1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # For people not in any ROI or when light is yellow, don't draw any box or label
            pass

red_light_dashers = set()
vehicles_passed_legally = set()
vehicles_in_bottom_roi = set()
vehicles_exited_bottom_roi = set()

def check_red_light_dasher(frame, vehicles_detected, traffic_light_color, rois_crosswalks, classNames):
    global red_light_dashers, vehicles_passed_legally, vehicles_in_bottom_roi, vehicles_exited_bottom_roi
    bottom_roi = rois_crosswalks[0]  # Bottom crosswalk

    for vehicle in vehicles_detected:
        vx1, vy1, vx2, vy2, conf, cls, track_id = vehicle

        if is_in_bottom_roi(vy1, vy2, bottom_roi):
            vehicles_in_bottom_roi.add(track_id)
        elif track_id in vehicles_in_bottom_roi:
            vehicles_exited_bottom_roi.add(track_id)
            vehicles_in_bottom_roi.remove(track_id)

            if traffic_light_color == 'Red' and track_id not in vehicles_passed_legally:
                # Vehicle exited while light was red and hadn't passed legally before
                red_light_dashers.add(track_id)
            elif traffic_light_color in ['Green', 'Yellow']:
                # Vehicle passed legally
                vehicles_passed_legally.add(track_id)

        if track_id in red_light_dashers and track_id in vehicles_exited_bottom_roi:
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 2)
            cv2.putText(frame, f'Dasher {conf}', (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 255, 255), 2)
            cv2.putText(frame, f'ID: {track_id}', (vx1, vy1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cvzone.cornerRect(frame, (vx1, vy1, vx2 - vx1, vy2 - vy1), l=5, rt=1)
            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, vx1), max(0, vy1)), scale=1, thickness=2)
            cv2.putText(frame, f'ID: {track_id}', (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def is_in_bottom_roi(vy1, vy2, bottom_roi):
    _, by, _, bh = bottom_roi
    return by <= vy2 and vy1 <= by + bh  # Vehicle is at least partially inside the ROI

def check_speeding(frame, vehicles_detected, speed_limit=45):
    for vehicle in vehicles_detected:
        vx1, vy1, vx2, vy2, conf, cls, track_id, red_light_dashing, speed = vehicle

        # Round speed to the nearest whole number
        speed = round(speed)

        # If speed is 1 m/s or lower, set it to 0 m/s
        if speed < 2:
            speed = 0

        # Display speed and speeding status
        if speed > speed_limit:
            cv2.putText(frame, f'Speeding: {speed} m/s', (vx1, vy2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # cv2.putText(frame, f'ID: {track_id}', (vx1, vy1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # cv2.putText(frame, f'Speed: {speed} m/s', (vx1, vy1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def calculate_time_from_frame(frame_number, start_time="06:00:27:0", fps=10):
    # Parse the start time
    start_hours, start_minutes, start_seconds, start_milliseconds = map(int, start_time.split(":"))

    # Calculate total milliseconds since start
    total_frames = frame_number
    milliseconds_per_frame = 1000 / fps  # ms per frame

    total_milliseconds = int(total_frames * milliseconds_per_frame)
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    total_minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(total_minutes, 60)

    # Add start time
    end_milliseconds = (start_milliseconds + milliseconds) % 1000
    carry_seconds = (start_milliseconds + milliseconds) // 1000

    end_seconds = (start_seconds + seconds + carry_seconds) % 60
    carry_minutes = (start_seconds + seconds + carry_seconds) // 60

    end_minutes = (start_minutes + minutes + carry_minutes) % 60
    carry_hours = (start_minutes + minutes + carry_minutes) // 60

    end_hours = start_hours + hours + carry_hours

    # Format the time
    end_time = f"{end_hours:02}:{end_minutes:02}:{end_seconds:02}.{end_milliseconds:03}"

    return end_time

if __name__ == "__main__":
    main()

