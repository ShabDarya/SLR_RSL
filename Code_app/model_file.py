import numpy as np
import torch
from class_model import GestureModel
# Параметры

SEQUENCE_LENGTH = 48
MAX_HANDS = 2
NUM_LANDMARKS = 21 #количество точек для каждой руки
NUM_FEATURES_PER_LANDMARK_HAND = 3 #количество координат для каждой точки
NUM_FEATURES = MAX_HANDS * NUM_LANDMARKS * NUM_FEATURES_PER_LANDMARK_HAND  #21 точка по 3 координаты

NUM_FEATURES_PER_LANDMARK = 2
LIPS_IDX = [61, 37, 0, 267, 291, 405, 17, 181] #точки губ
NUM_FEATURES_FACE = len(LIPS_IDX)

POSE_IDX = [16, 14, 12, 11, 13, 15, 0] #точки рук + нос
NUM_FEATURES_POSE = len(POSE_IDX)

NUM_FEATURES_ALL = NUM_FEATURES + NUM_FEATURES_FACE * NUM_FEATURES_PER_LANDMARK + NUM_FEATURES_POSE * NUM_FEATURES_PER_LANDMARK

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_model():
    model = GestureModel(num_classes=1001, d_model=384, dropout=0.3)
    checkpoint_path = r"best_model_epoch_266_top1_33.64.pth"
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    return model 
    

def get_landmarks(results_hand, results_face, results_pose):
    
    landmarks = []

    if results_hand.hand_landmarks:
        landmarks_hands = []
        hand_idx = 0
        for hand in results_hand.hand_landmarks[:MAX_HANDS]:
            hand_idx += 1
            for lm in hand:
                landmarks_hands.extend([lm.x, lm.y, lm.z])

        while len(landmarks_hands) < NUM_FEATURES:
            landmarks_hands.extend([np.nan] * NUM_LANDMARKS * NUM_FEATURES_PER_LANDMARK_HAND)
        landmarks.extend(landmarks_hands)
    else:
        landmarks.extend([np.nan] * NUM_FEATURES)

    if results_face.face_landmarks:
        for idx in LIPS_IDX:
            lm = results_face.face_landmarks[0][idx]
            landmarks.extend([lm.x, lm.y])
    else:
        landmarks.extend([np.nan] * NUM_FEATURES_FACE * NUM_FEATURES_PER_LANDMARK)


    if results_pose.pose_landmarks:
        for idx in POSE_IDX:
            lm = results_pose.pose_landmarks[0][idx]
            landmarks.extend([lm.x, lm.y])
    else:
        landmarks.extend([np.nan] * NUM_FEATURES_POSE * NUM_FEATURES_PER_LANDMARK)

    if len(landmarks) != NUM_FEATURES_ALL:
        print("WRONG FRAME")
        print("len(landmarks):", len(landmarks))
        print("expected:", NUM_FEATURES_ALL)
        raise ValueError("Feature length mismatch")

    return [0 if np.isnan(x) else x for x in landmarks]

def get_gesture(model, landmarks):
   landmarks_float32 = np.array(landmarks, dtype=np.float32)

   with torch.no_grad():
      outputs = model(torch.tensor(landmarks_float32).unsqueeze(0))

   res = model.idx2text(outputs)
   return res
