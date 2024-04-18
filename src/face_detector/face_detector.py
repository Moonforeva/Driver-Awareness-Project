import mediapipe as mp
import time

class FaceDetector_and_Result_Video():
    def __init__(self):
        self.result = mp.tasks.vision.FaceDetectorResult
        self.face_detector = mp.tasks.vision.FaceDetector
        self.getResults()
    
    def getResults(self):
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options = mp.tasks.BaseOptions(model_asset_path='./face_detector/mediapipe_face/blaze_face_short_range.tflite'),
            running_mode = mp.tasks.vision.RunningMode.VIDEO,
            min_detection_confidence = 0.5,
            min_suppression_threshold = 0.3,)
        
        self.face_detector = self.face_detector.create_from_options(options)

    def detect_for_video(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.face_detector.detect_for_video(image = mp_image, timestamp_ms = int(time.time() * 1000))

    def close(self):
        self.face_detector.close()

class FaceLandmarks_and_Result_Video():
    def __init__(self):
        self.result = mp.tasks.vision.FaceLandmarkerResult
        self.face_landmarkers = mp.tasks.vision.FaceLandmarker
        self.getResults()
    
    def getResults(self):      
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options = mp.tasks.BaseOptions(model_asset_path='./face_detector/mediapipe_face/face_landmarker.task'),
            running_mode = mp.tasks.vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,)
        
        self.face_landmarkers = self.face_landmarkers.create_from_options(options)

    def detect_for_video(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.face_landmarkers.detect_for_video(image = mp_image, timestamp_ms = int(time.time() * 1000))
    
    def close(self):
        self.face_landmarkers.close()

class FaceDetector_and_Result_LiveStream():
    def __init__(self):
        self.result = mp.tasks.vision.FaceDetectorResult
        self.face_detector = mp.tasks.vision.FaceDetector
        self.getResults()
    
    def getResults(self):
        def update_result(result: mp.tasks.vision.FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result
        
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options = mp.tasks.BaseOptions(model_asset_path='./face_detector/mediapipe_face/blaze_face_short_range.tflite'),
            running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM,
            min_detection_confidence = 0.5,
            min_suppression_threshold = 0.3,
            result_callback=update_result)
        
        self.face_detector = self.face_detector.create_from_options(options)

    def detect_async(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.face_detector.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))
    
    def close(self):
        self.face_detector.close()

class FaceLandmarks_and_Result_LiveStream():
    def __init__(self):
        self.result = mp.tasks.vision.FaceLandmarkerResult
        self.face_landmarkers = mp.tasks.vision.FaceLandmarker
        self.getResults()
    
    def getResults(self):
        def update_result(result: mp.tasks.vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result
        
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options = mp.tasks.BaseOptions(model_asset_path='./face_detector/mediapipe_face/face_landmarker.task'),
            running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            result_callback=update_result)
        
        self.face_landmarkers = self.face_landmarkers.create_from_options(options)

    def detect_async(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.face_landmarkers.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))
    
    def close(self):
        self.face_landmarkers.close()