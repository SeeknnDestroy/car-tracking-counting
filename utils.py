import cv2


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
