from inference import inference
from visualization import save_to_csv, visualize_data
from config import model_path, video_path, export_path, device, imgsz

from ultralytics import YOLO


def main() -> None:
    """
    Main execution function to run the car tracking and counting project.
    """
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Run inference and save the annotated video
    inference_results = inference(model, video_path=video_path, export_path=export_path, device=device, imgsz=imgsz, save=True)

    # Save the car tracking data
    data_path = "car_data.csv"
    save_to_csv(inference_results, data_path)

    # Generate visualizations
    visualize_data(data_path)


if __name__ == "__main__":
    main()
