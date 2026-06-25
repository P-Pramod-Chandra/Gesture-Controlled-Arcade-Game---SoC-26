import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def get_landmark_list(frame, hand_landmarks):
    h, w, _ = frame.shape
    lm_list = []
    for idx, lm in enumerate(hand_landmarks.landmark):
        cx, cy = int(lm.x * w), int(lm.y * h)   # normalized → pixels
        lm_list.append([idx, cx, cy])
    return lm_list


def fingers_up(lm_list):
    """Returns a list of 5 ints [thumb, index, middle, ring, pinky], 1 = up."""
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb: compare x (sideways). Assumes a right hand on a mirrored feed.
    fingers.append(1 if lm_list[4][1] > lm_list[3][1] else 0)

    # Other four fingers: compare y (tip above PIP joint = up)
    for tip in tips[1:]:
        fingers.append(1 if lm_list[tip][2] < lm_list[tip - 2][2] else 0)

    return fingers


def classify_gesture(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return "FIST"
    if fingers == [1, 1, 1, 1, 1]:
        return "OPEN PALM"
    if fingers == [0, 1, 0, 0, 0]:
        return "POINTING"
    if fingers == [0, 1, 1, 0, 0]:
        return "PEACE"
    if fingers == [1, 1, 0, 0, 1]:
        return "GO WEB"
    return "UNKNOWN"


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,             # one hand is plenty for this project
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)                       # mirror, feels natural
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # BGR → RGB for MediaPipe
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm_list = get_landmark_list(frame, hand_landmarks)
            if lm_list:
                fingers = fingers_up(lm_list)
                gesture = classify_gesture(fingers)
                print(fingers, gesture)
                cv2.putText(frame, gesture, (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()