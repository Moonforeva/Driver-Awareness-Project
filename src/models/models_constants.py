import torchvision.transforms as transforms

GAZE_CLASSES = ['Eyes Closed',
 'Forward',
 'Lap',
 'Left Mirror',
 'Radio',
 'Rearview',
 'Right Mirror',
 'Shoulder',
 'Speedometer']

DRIVER_DISTRACT_CLASSES = ["safe driving",
                           "texting-right",
                           "talking on the phone - right",
                           "texting-left",
                           "talking on the phone - left",
                           "operating the radio",
                           "drinking",
                           "reaching behind",
                           "hair and makeup",
                           "talking to passenger"]


BATCH_SIZE = 4
EPOCHS = 2

GAZE_MODEL_SAVE_PATH = "./models/eye_gaze_classifier.pth"
DRIVER_MODEL_SAVE_PATH = "./models/drive_distract_net.pth"

IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Example resize, adjust as needed
    transforms.ToTensor(),
])