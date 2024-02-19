import cv2
from datetime import datetime


def initialize_video_writer(export_path: str, frame_width: int, frame_height: int) -> cv2.VideoWriter:
    """
    Initialize and return a video writer object.

    Parameters:
        export_path (str): The path where the video will be saved.
        frame_width (int): The width of the video frames.
        frame_height (int): The height of the video frames.

    Returns:
        cv2.VideoWriter: The initialized video writer.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_rate = 30
    return cv2.VideoWriter(export_path, fourcc, frame_rate, (frame_width, frame_height))


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
