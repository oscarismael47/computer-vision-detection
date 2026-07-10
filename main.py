# STEP 1: Import the necessary modules.
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
FONT_SIZE = 2
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles
LANDMARK_DRAWING_SPEC = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=2)
CONNECTION_DRAWING_SPEC = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=5, circle_radius=2)

def draw_landmarks_on_image(bgr_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(bgr_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      LANDMARK_DRAWING_SPEC,
      CONNECTION_DRAWING_SPEC)

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

# STEP 2: Create a HandLandmarker object for video processing.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input video.
capture = cv2.VideoCapture("test_1.MP4")
if not capture.isOpened():
  raise FileNotFoundError("Could not open test_1.MP4")

video_fps = capture.get(cv2.CAP_PROP_FPS)
if not video_fps or video_fps <= 0:
  video_fps = 30.0

frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = "annotated_test_1.mp4"
video_writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    video_fps,
    (frame_width, frame_height))

if not video_writer.isOpened():
  capture.release()
  raise RuntimeError(f"Could not open video writer for {output_path}")

frame_index = 0

# STEP 4: Detect hand landmarks from each video frame.
while True:
  success, frame_bgr = capture.read()
  if not success:
    break

  frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
  mp_frame = mp.Image(
      image_format=mp.ImageFormat.SRGB,
      data=frame_rgb)

  timestamp_ms = int(frame_index * 1000 / video_fps)
  detection_result = detector.detect_for_video(mp_frame, timestamp_ms)

  # STEP 5: Visualize the detection result on the current frame.
  annotated_frame = draw_landmarks_on_image(frame_bgr, detection_result)
  video_writer.write(annotated_frame)
  cv2.imshow('Annotated Video', annotated_frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  frame_index += 1

capture.release()
video_writer.release()
cv2.destroyAllWindows()