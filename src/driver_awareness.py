import mediapipe as mp
import cv2
import numpy as np
import torch
import ffmpegcv
import argparse

from face_detector.face_detector import FaceDetector_and_Result_Video, FaceLandmarks_and_Result_Video
from face_detector.face_detector import FaceDetector_and_Result_LiveStream, FaceLandmarks_and_Result_LiveStream
from models.driver_distract import DriverDistractNet
from models.gaze_classification_net import GazeClassificationNet
from models.models_constants import *

def draw_face_bbox(rgb_image, detection_result: mp.tasks.vision.FaceDetectorResult, model):
    try:
        if detection_result.detections == []:
            return rgb_image
        else:
            face_detection_list = detection_result.detections
            annotated_image = np.copy(rgb_image)

            for detection in face_detection_list:
                bbox = detection.bounding_box
                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

                eye_start = bbox.origin_x , bbox.origin_y + int(0.1 * bbox.height)
                eye_end = bbox.origin_x + bbox.width, bbox.origin_y + int(0.4 * bbox.height)

                eye_frame = annotated_image[bbox.origin_y + int(0.4*bbox.origin_y):bbox.origin_y+int(bbox.height-0.6*bbox.height),bbox.origin_x:bbox.origin_x+bbox.width]
                eye_frame_tensor = IMAGE_TRANSFORM(eye_frame)
                eye_frame_tensor = eye_frame_tensor.unsqueeze(0)
                with torch.no_grad():
                    outputs = model(eye_frame_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    predicted_text = f"{GAZE_CLASSES[predicted.item()]}"
                annotated_image = cv2.putText(annotated_image,predicted_text,(50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
                annotated_image = cv2.rectangle(annotated_image, start_point, end_point, (0,0,255), 3)
                annotated_image = cv2.rectangle(annotated_image, eye_start, eye_end, (0,255,0), 3)
            return annotated_image
    except Exception as e:
        # print("Error in draw_face_bbox:", e)
        return rgb_image    

def face_angle(rgb_image,landmarks_results:mp.tasks.vision.FaceLandmarkerResult):
    img_h, img_w = rgb_image.shape[0], rgb_image.shape[1]
    face_3d = []
    face_2d = []
    try:
        if landmarks_results.face_landmarks == []:
            return rgb_image
        else:
            face_landmarks_list = landmarks_results.face_landmarks
            annotated_image = np.copy(rgb_image)

            for face_landmarks in face_landmarks_list:
                for idx, lm in enumerate(face_landmarks):
                    if idx == 33 or idx == 263 or idx == 4 or idx == 186 or idx == 410 or idx == 152:
                        x, y = int(lm.x * img_w) , int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = int(img_w)
                center = (int(img_w/2), int(img_h/2))
                camera_matrix = np.array(
                    [[focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]] , dtype=np.double
                )

                dist_coeffs = np.zeros((4,1)) # assume no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(face_3d, face_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

                rotation_matrix, jac = cv2.Rodrigues(rotation_vector)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)
                x = angles[0] * 360
                y = angles[1] * 360
                angle_text = f"x:{x:.4f} y:{y:.4f}"
                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([0.0, 0.0, 1000.0]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

                for p in face_2d:
                    annotated_image = cv2.circle(annotated_image, (int(p[0]), int(p[1])), 3, (0,0,255),-1)
                p1 = (int(face_2d[0][0]), int(face_2d[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                annotated_image = cv2.line(annotated_image, p1, p2, (255,0,0), 2)
                annotated_image = cv2.putText(annotated_image,angle_text,(50,50),cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
            return annotated_image
    except Exception as e:
        # print("Error in face_angles:", e)
        return rgb_image 

def driver_distract(rgb_image,model):
    annotated_image = np.zeros_like(rgb_image)
    with torch.no_grad():
        driver_frame_tensor = IMAGE_TRANSFORM(rgb_image)
        driver_frame_tensor = driver_frame_tensor.unsqueeze(0)
        outputs = model(driver_frame_tensor)
        _, predicted = torch.max(outputs.data, 1)
        predicted_text = f"{DRIVER_DISTRACT_CLASSES[predicted.item()]}"
    annotated_image = cv2.putText(annotated_image,predicted_text,(50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    return annotated_image

def multiple_frame(*frames):
    multiframes = np.hstack(frames)
    return multiframes

def masking(frame):
    blank = np.zeros_like(frame, dtype=np.uint8)
    width = blank.shape[1]
    height = blank.shape[0]
    blank[:, :int(width/2)] = [255, 255, 255]
    masked = cv2.bitwise_and(frame,blank)
    return masked

def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-v","--video", help="path for video input, if None automatically is webcam")
    args = vars(ap.parse_args())
    # LOADING MODELS FOR DRIVER DISTRACTION AND GAZE CLASSIFICATION
    driver_distract_model = DriverDistractNet()
    driver_distract_model.load_state_dict(torch.load(DRIVER_MODEL_SAVE_PATH))
    driver_distract_model.eval()

    gaze_model = GazeClassificationNet()
    gaze_model.load_state_dict(torch.load(GAZE_MODEL_SAVE_PATH))
    gaze_model.eval()

    # INITIALIZE VIDEOCAPTURE AND FACE LANDMARKS DETECTOR
    
    def video_source(video_file = None):
        if video_file:
            cap = cv2.VideoCapture(video_file)
            face_detectors = FaceDetector_and_Result_Video()
            face_landmarks = FaceLandmarks_and_Result_Video()
            out = ffmpegcv.VideoWriter("./data/output/video_result.mp4")
            while True:
                # pull frame
                ret, frame_ori = cap.read()
                frame_ori = cv2.resize(frame_ori, (640,360))
                frame_ori = masking(frame_ori)
                # mirror frame
                face_detectors.detect_for_video(frame_ori)
                face_landmarks.detect_for_video(frame_ori)
                results_face_frame = draw_face_bbox(frame_ori, face_detectors.result, gaze_model)
                driver_distract_frame = driver_distract(frame_ori, driver_distract_model)
                face_angle_frame = face_angle(frame_ori,face_landmarks.result)
                multiframes = multiple_frame(results_face_frame,face_angle_frame,driver_distract_frame)
                # display frame
                # print(multiframes.shape)
                out.write(multiframes)
                cv2.imshow('frame',multiframes)
                if cv2.waitKey(1) == ord('q'):
                    break

            # release everything
            face_detectors.close()
            face_landmarks.close()
            out.release()
            cap.release()
            cv2.destroyAllWindows()
        else:
            cap = cv2.VideoCapture(0)
            face_detectors = FaceDetector_and_Result_LiveStream()
            face_landmarks = FaceLandmarks_and_Result_LiveStream()
            out = ffmpegcv.VideoWriter("./data/output/livestream_result.mp4")
            while True:
                # pull frame
                ret, frame_ori = cap.read()
                # mirror frame
                frame_ori = cv2.flip(frame_ori, 1)
                face_detectors.detect_async(frame_ori)
                face_landmarks.detect_async(frame_ori)
                results_face_frame = draw_face_bbox(frame_ori, face_detectors.result, gaze_model)
                driver_distract_frame = driver_distract(frame_ori, driver_distract_model)
                face_angle_frame = face_angle(frame_ori,face_landmarks.result)
                multiframes = multiple_frame(results_face_frame,face_angle_frame,driver_distract_frame)
                # display frame
                out.write(multiframes)
                cv2.imshow('frame',multiframes)
                if cv2.waitKey(1) == ord('q'):
                    break

            # release everything
            face_detectors.close()
            face_landmarks.close() 
            out.release()
            cap.release()
            cv2.destroyAllWindows()

    video_source(args["video"])

if __name__ == "__main__":
    main()