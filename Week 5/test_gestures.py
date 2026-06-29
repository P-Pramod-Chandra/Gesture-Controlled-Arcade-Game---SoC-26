import cv2
import mediapipe as mp
import gestures   # your module

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = "NONE"
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        label = "Right"
        if results.multi_handedness:
            label = results.multi_handedness[0].classification[0].label

        h, w, _ = frame.shape
        lm_list = [[i, int(lm.x * w), int(lm.y * h)]
                   for i, lm in enumerate(hand.landmark)]

        gesture = gestures.classify(lm_list, label)
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, gesture, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
    cv2.imshow("Gesture Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()