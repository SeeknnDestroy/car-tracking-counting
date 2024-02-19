
# Configuration for line positions and frame dimensions
line_position_horizontal: int = 800
line_position_perpendicular: int = 1200
frame_width: int = 1920
frame_height: int = 1080

# Configuration for the model and inference
device: str = 'cuda:0'  # Use 'cpu' for CPU
imgsz: tuple[int, int] = (384, 640)
model_path: str = 'yolov8m.pt'

# Configuration for the input video and export path
video_path: str = 'input.mp4'
export_path: str = 'output.mp4'
