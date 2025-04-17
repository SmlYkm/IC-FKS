from ultralytics import YOLO
import supervision as sv
import numpy as np
import os
import re
import cv2

def get_next_run_filename(folder="runs", base_name="tracked_output", extension=".mp4"):
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    
    # Pattern to find existing files
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+){re.escape(extension)}")
    max_n = 0
    
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            n = int(match.group(1))
            max_n = max(max_n, n)
    
    next_n = max_n + 1
    return os.path.join(folder, f"{base_name}_{next_n}{extension}")

model = YOLO("yolov8s.pt")
model.to("cuda")
print("Model device:", model.device)

SELECTED_CLASS_NAMES = ["person", "sports ball"]
CLASS_NAMES_DICT = model.names
# class ids matching the class names we have chosen
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name
    in SELECTED_CLASS_NAMES
]

SOURCE_VIDEO_PATH = "video.mp4"
TARGET_VIDEO_PATH = get_next_run_filename(
    folder="runs", 
    base_name="tracked_output", 
    extension=".mp4"
)

# create BYTETracker instance
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=30,
    minimum_consecutive_frames=3
)

# create instance of BoxAnnotator, LabelAnnotator, and TraceAnnotator
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# define call back function to be used in video processing
def process_frame(frame: np.ndarray) -> np.ndarray:
    # model prediction on single frame and conversion to supervision Detections
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    # only consider class id from selected_classes define above
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    # tracking detections
    detections = byte_tracker.update_with_detections(detections)
    labels = [
        f"#{tracker_id} {model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id, tracker_id
        in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]
    annotated_frame = frame.copy()
    
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    return annotated_frame

def process_video(
    source_path: str, 
    target_path: str, 
    start_frame: int = 0,
    end_frame: int = None
) -> None:
    # Open the source video
    src_video = cv2.VideoCapture(source_path)

    # Get video properties
    width = int(src_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(src_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(src_video.get(cv2.CAP_PROP_FPS))
    total_frames = int(src_video.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        total_frames = total_frames - start_frame
    else:
        total_frames = end_frame - start_frame

    # Set the starting frame
    src_video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_video = cv2.VideoWriter(
        target_path, 
        cv2.VideoWriter_fourcc(*"mp4v"), 
        fps, 
        (width, height)
    )

    # Process frames
    for i in range(total_frames):
        progress =  i / total_frames * 100
        print(f"Processing frame {i}/{total_frames} ({progress:.2f}%)", end="\r")

        ret, frame = src_video.read()
        if not ret:
            print("End of video stream")
            break

        # Process frame
        frame = process_frame(frame)
        output_video.write(frame)

    # Release resources
    src_video.release()
    output_video.release()

process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    start_frame=1000,
    end_frame=2000
)