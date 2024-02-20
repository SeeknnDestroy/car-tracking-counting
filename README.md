# Car Tracking and Counting

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SeeknnDestroy/car-tracking-counting/blob/main/car_tracking.ipynb)

## Overview
This project aims to accurately track and count vehicles in video footage using advanced object detection and tracking technologies, specifically employing YOLOv8 for detection and ByteTrack for tracking. The ultimate goal is to assign unique IDs to vehicles, count them based on their direction, convert the model to ONNX format for performance evaluation, and visualize the tracking data.

## Results
Key achievements include:
- **Annotated Video & Car Counts**: Vehicles being tracked with bounding boxes and IDs, alongside counts of vehicles moving in each direction. Full inference video link [here](https://drive.google.com/drive/folders/1XqBpGF5KUr1G5-v0Ag5c2VG8S5FyNBQu?usp=sharing), and a snapshot below:
![annotated_video](https://github.com/SeeknnDestroy/car-tracking-counting/assets/44926076/71c9dcb0-60ad-4cb7-99fe-509322bc2dde)

- **Model Conversion & Performance Comparison**: Successful conversion of the detection model to ONNX format, with a comparative analysis of inference speeds between the original and ONNX models on a CPU.
    ```
    ONNX Inference Time: 802.90 seconds
    PyTorch Inference Time: 846.92 seconds
    ```

- **Data Visualization**: Plots generated from the CSV data, offering insights into the traffic flow and vehicle counts over time.
![number_of_cars](https://github.com/SeeknnDestroy/car-tracking-counting/assets/44926076/a47785ab-36fe-4a33-836d-c0c8033d987b)
![total_count_of_cars](https://github.com/SeeknnDestroy/car-tracking-counting/assets/44926076/b55951c3-aa57-41a0-ba68-0ef4f0094d31)


## Reproducing Results
To reproduce the project's results, you can either follow the interactive steps in the `car_tracking.ipynb` Jupyter notebook or directly execute the `main.py` script. Both methods will process the video, annotate it, and produce a CSV file with the vehicle counts, in addition to generating visualizations and annotated video.

### Option 1: Jupyter Notebook
1. Launch Jupyter Notebook and open `car_tracking.ipynb`.
2. Run the cells sequentially.

### Option 2: Python Script
1. Ensure your video is in the project directory.

2. Open `config.py` and set the `video_path` variable to the video's file name.

3. Execute:

```bash
python main.py
```

## Setup
Follow these steps to set up your environment:
1. Clone the repository:

    ```bash
    git clone https://github.com/SeeknnDestroy/car-tracking-counting.git
    cd car-tracking-counting
    ```

2. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

3. Create a Python 3.10 env and activate it:

    ```bash
    conda create -n car-tracking python=3.10
    conda activate car-tracking
    ```

4. Install the required packages via `pip install -r requirements.txt`.

5. Modify `config.py` for different model settings or file paths.

6. Run the python script: `python main.py`.

Ensure your system's hardware is compatible with YOLOv8, especially for GPU usage.

## Future Improvement Ideas
- **BoT-SORT**: Using `Bot-SORT` instead of ByteTrack for tracking boosts performance significantly, especially for real-time applications. But has accuracy trade-offs.
- **Image Resizing**: Further optimize the model by resizing images to a smaller resolution, which can improve inference speed.
- **Using Smaller Models**: Employing smaller models like `YOLOv8s` or `YOLOv8n` is another way to improve performance. But again, this comes with a trade-off in accuracy.
- **Use Real FPS**: The current implementation uses a fixed FPS assumption of 30, which can be replaced with real-time FPS calculations (for example, my inference videos had 29 FPS) for more accurate time-based data. (This is especially important for real-time applications.)

## Legacy Code Reference

Included in this repository is a legacy module, `legacy_onnx_detector.py`, initially developed to explore ONNX model integration for object detection. While it showcases sophisticated handling of ONNX runtime and preprocessing techniques, this module was ultimately not utilized in the project's final iteration due to its performance efficiency compared to our selected approach. It remains part of the codebase to demonstrate the exploration of diverse solutions in the development process and may serve as a reference for future projects.
