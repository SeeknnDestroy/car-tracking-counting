from datetime import datetime, timedelta
from typing import List, Dict

from config import (
    line_position_horizontal,
    line_position_perpendicular,
    START_POINT_HORIZONTAL,
    END_POINT_HORIZONTAL,
    START_POINT_PERPENDICULAR,
    END_POINT_PERPENDICULAR)
from utils import initialize_video_writer, update_car_count_and_record_state

import cv2
from ultralytics import YOLO


def inference(model: YOLO, video_path: str, export_path: str, device: str = 'cpu', imgsz=(640,640), save: bool = True) -> List[Dict]:
    """
    Run inference on the input video and save the annotated video if specified.

    Parameters:
        model (YOLO): The YOLO model used for tracking.
        video_path (str): The path to the input video.
        export_path (str): The path to save the annotated video.
        device (str, optional): The device to run the inference on. Defaults to 'cpu'.
        imgsz (tuple, optional): The size of the input image. Defaults to (640, 640).
        save (bool, optional): Whether to save the annotated video. Defaults to True.

    Returns:
        List[Dict]: A list of state changes with timestamps.
    """
    cap = cv2.VideoCapture(video_path)
    # Ensure the frame dimensions are integers
    target_height, target_width = imgsz
    out = initialize_video_writer(export_path, target_width, target_height)

    # Initialize simulation variables
    start_time = datetime.strptime("19.02.2024 13:50:00", "%d.%m.%Y %H:%M:%S")  # Simulated start time
    frame_rate = 30  # Assuming 30 FPS for the video
    time_per_frame = timedelta(seconds=1/frame_rate)
    current_time = start_time

    # Initialize tracking variables
    car_positions = {}
    car_counts = {'UP': 0, 'DOWN': 0, 'LEFT': 0, 'RIGHT': 0}
    already_counted = {}    # Tracks whether a car has been counted to prevent double counting
    state_changes = []  # To record state changes with timestamps

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        # resize frame to the specified size
        frame = cv2.resize(frame, (target_width, target_height))

        # Draw the horizontal and perpendicular lines
        cv2.line(frame, START_POINT_HORIZONTAL, END_POINT_HORIZONTAL, (0, 255, 0), 2)
        cv2.line(frame, START_POINT_PERPENDICULAR, END_POINT_PERPENDICULAR, (255, 0, 0), 2)

        # Run YOLOv8 tracking on the frame
        results = model.track(frame, classes=[2, 7], persist=True, device=device, imgsz=imgsz, conf=0.1, iou=0.5, tracker="bytetrack.yaml")   # Focusing on cars (class 2) and trucks (class 7)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                center = (int(x), int(y))
                bbox_color = (255, 0, 0)  # Default color

                # Determine if the car has crossed the lines and update counts
                if track_id in car_positions:
                    prev_center = car_positions[track_id]
                    # Horizontal line crossing logic
                    if prev_center[1] < line_position_horizontal <= center[1]:
                        update_car_count_and_record_state(track_id, 'DOWN', car_counts, already_counted, current_time, state_changes)
                        bbox_color = (0, 255, 0)
                    elif prev_center[1] > line_position_horizontal >= center[1]:
                        update_car_count_and_record_state(track_id, 'UP', car_counts, already_counted, current_time, state_changes)
                        bbox_color = (0, 255, 0)
                    # Perpendicular line crossing logic
                    if prev_center[0] < line_position_perpendicular <= center[0]:
                        update_car_count_and_record_state(track_id, 'RIGHT', car_counts, already_counted, current_time, state_changes)
                        bbox_color = (255, 0, 0)
                    elif prev_center[0] > line_position_perpendicular >= center[0]:
                        update_car_count_and_record_state(track_id, 'LEFT', car_counts, already_counted, current_time, state_changes)
                        bbox_color = (255, 0, 0)

                # Update the car's current position
                car_positions[track_id] = center

                # Draw bounding box and track ID
                cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), bbox_color, 2)
                cv2.putText(frame, f"ID: {track_id}", (int(x - w / 2), int(y - h / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

        # Draw the car counts and current time
        draw_car_counts_and_time(frame, car_counts, current_time, frame_height)

        # Increment the simulated time
        current_time += time_per_frame

        # Write the frame with annotations to the output video
        if save:
            out.write(frame)

    # Release the video capture and writer
    cap.release()
    out.release()

    return state_changes


def update_car_count_and_record_state(
        track_id: int,
        direction: str,
        car_counts: dict,
        already_counted: dict,
        current_time: datetime,
        state_changes: list):
    """
    Update the car count for the specified direction if the car hasn't been counted in that direction yet and record the state change.

    Parameters:
        track_id (int): The ID of the car being tracked.
        direction (str): The direction in which the car is moving.
        car_counts (dict): A dictionary containing the car counts for each direction.
        already_counted (dict): A dictionary containing the directions in which each car has already been counted.
        current_time (datetime): The current timestamp.
        state_changes (list): A list of state changes.

    Returns:
        None
    """
    if track_id not in already_counted or already_counted[track_id] != direction:
        car_counts[direction] += 1
        already_counted[track_id] = direction
        # Record the state change with a precise timestamp
        state_changes.append(
            {
                'car_id': track_id,
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                'state': direction
            }
        )


def draw_car_counts_and_time(frame, car_counts: Dict, current_time: datetime, frame_height: int) -> None:
    """
    Draw the car counts and current time on the frame.

    Parameters:
        frame (MatLike): The frame to draw on.
        car_counts (Dict): The car counts for each direction.
        current_time (datetime): The current timestamp.
        frame_height (int): The height of the frame.

    Returns:
        None
    """
    cv2.putText(frame, f"Up: {car_counts['UP']} Down: {car_counts['DOWN']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Left: {car_counts['LEFT']} Right: {car_counts['RIGHT']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
