import cv2 
import mediapipe as mp 
import pyautogui

mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils 

cap = cv2.VideoCapture(0) 

def is_gun_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    is_gun = index_tip.y < middle_finger_tip.y and \
             abs(index_tip.x - thumb_tip.x) > 0.1

    return is_gun

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        success, image = cap.read() 
        if not success: 
            continue 

        # Flip image for correct handedness
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image) 
        
        # Draw hand landmarks 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        if results.multi_hand_landmarks: 
            for hand_landmarks in results.multi_hand_landmarks: 
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS) 

                # Check for gun gesture
                if is_gun_gesture(hand_landmarks):
                    print("Gun Gesture Recognized")
                    #pyautogui.press('space')

        # Display image
        cv2.imshow('MediaPipe Hands', image)
        
        # Exit on ESC key
        if cv2.waitKey(5) & 0xFF == 27: 
            break 

cap.release()
cv2.destroyAllWindows()
