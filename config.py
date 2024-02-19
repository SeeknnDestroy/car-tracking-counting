
# Configuration for line positions
line_position_horizontal: int = 240
line_position_perpendicular: int = 400
START_POINT_HORIZONTAL: tuple[int, int] = (0, line_position_horizontal)
END_POINT_HORIZONTAL: tuple[int, int] = (640, line_position_horizontal)
START_POINT_PERPENDICULAR: tuple[int, int] = (line_position_perpendicular, 0)
END_POINT_PERPENDICULAR: tuple[int, int] = (line_position_perpendicular, 384)

# Configuration for the model and inference
device: str = 'cuda:0'  # Use 'cpu' for CPU
imgsz: tuple[int, int] = (384, 640)
model_path: str = 'yolov8m.pt'

# Configuration for the input video and export path
video_path: str = 'input.mp4'
export_path: str = 'output.mp4'
