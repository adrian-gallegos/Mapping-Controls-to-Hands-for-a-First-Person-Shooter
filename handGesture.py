import cv2 
import mediapipe as mp 
import pyautogui

mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils 

cap = cv2.VideoCapture(0) 

def equality(num, num2):
    return abs(num - num2) < .1

def is_gun_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    is_gun = index_tip.y < middle_finger_tip.y and thumb_tip.y < index_tip.y \
        and abs(index_tip.x - thumb_tip.x) > 0.1

    return is_gun

def is_one(hand_landmarks):
    landmarks = hand_landmarks.landmark
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]

    middle_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    ring_dip = landmarks[mp_hands.HandLandmark.RING_FINGER_DIP]
    pinky_dip = landmarks[mp_hands.HandLandmark.PINKY_DIP]
    
    is_one = index_tip.y < index_dip.y and index_dip.y < index_pip.y and index_pip.y < index_mcp.y \
        and index_pip.y < middle_pip.y and index_pip.y < ring_pip.y and index_pip.y < pinky_pip.y and index_pip.y < thumb_tip.y \
        and middle_pip.y < middle_dip.y and ring_pip.y < ring_dip.y and pinky_pip.y < pinky_dip.y and index_pip.x < thumb_tip.x
    return is_one

def is_two(hand_landmarks):
    landmarks = hand_landmarks.landmark
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]

    middle_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    ring_dip = landmarks[mp_hands.HandLandmark.RING_FINGER_DIP]
    pinky_dip = landmarks[mp_hands.HandLandmark.PINKY_DIP]

    
    is_two = index_tip.y < index_dip.y and index_dip.y < index_pip.y and index_pip.y < index_mcp.y \
        and middle_tip.y < middle_dip.y and middle_dip.y < middle_pip.y and middle_pip.y < middle_mcp.y \
        and index_pip.y < ring_pip.y and index_pip.y < pinky_pip.y and index_pip.y < thumb_tip.y \
        and ring_pip.y < ring_dip.y and pinky_pip.y < pinky_dip.y
    return is_two

def is_three(hand_landmarks):
    landmarks = hand_landmarks.landmark
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]

    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    middle_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    ring_dip = landmarks[mp_hands.HandLandmark.RING_FINGER_DIP]
    pinky_dip = landmarks[mp_hands.HandLandmark.PINKY_DIP]

    is_three = index_tip.y < index_dip.y and index_dip.y < index_pip.y and index_pip.y < index_mcp.y \
        and middle_tip.y < middle_dip.y and middle_dip.y < middle_pip.y and middle_pip.y < middle_mcp.y \
        and ring_tip.y < ring_dip.y and ring_dip.y < ring_pip.y and ring_pip.y < ring_mcp.y \
        and index_pip.y < pinky_pip.y and index_pip.y < thumb_tip.y \
        and pinky_pip.y < pinky_dip.y

    return is_three


def is_four(hand_landmarks):
    landmarks = hand_landmarks.landmark
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]

    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]

    middle_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    ring_dip = landmarks[mp_hands.HandLandmark.RING_FINGER_DIP]
    pinky_dip = landmarks[mp_hands.HandLandmark.PINKY_DIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]
    


    is_four = index_tip.y < index_dip.y and index_dip.y < index_pip.y and index_pip.y < index_mcp.y \
        and middle_tip.y < middle_dip.y and middle_dip.y < middle_pip.y and middle_pip.y < middle_mcp.y \
        and ring_tip.y < ring_dip.y and ring_dip.y < ring_pip.y and ring_pip.y < ring_mcp.y \
        and pinky_tip.y < pinky_dip.y and pinky_dip.y < pinky_pip.y and pinky_pip.y < pinky_mcp.y \
        and index_pip.y < thumb_tip.y and thumb_tip.x > thumb_ip.x
    return is_four


def is_five(hand_landmarks):
    landmarks = hand_landmarks.landmark

    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]

    thumb_cmc = landmarks[mp_hands.HandLandmark.THUMB_CMC]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]

    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]

    middle_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    ring_dip = landmarks[mp_hands.HandLandmark.RING_FINGER_DIP]
    pinky_dip = landmarks[mp_hands.HandLandmark.PINKY_DIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    is_five =  index_tip.y < index_dip.y and index_dip.y < index_pip.y and index_pip.y < index_mcp.y \
        and middle_tip.y < middle_dip.y and middle_dip.y < middle_pip.y and middle_pip.y < middle_mcp.y \
        and ring_tip.y < ring_dip.y and ring_dip.y < ring_pip.y and ring_pip.y < ring_mcp.y \
        and pinky_tip.y < pinky_dip.y and pinky_dip.y < pinky_pip.y and pinky_pip.y < pinky_mcp.y \
        and thumb_cmc.y > thumb_mcp.y and thumb_mcp.y > thumb_ip.y and thumb_ip.y > thumb_tip.y and thumb_tip.x < thumb_ip.x
    return is_five

def is_six(hand_landmarks):
    landmarks = hand_landmarks.landmark
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    thumb_cmc = landmarks[mp_hands.HandLandmark.THUMB_CMC]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]

    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]
    
    is_six = equality(index_pip.y, middle_pip.y) \
        and thumb_cmc.y > thumb_mcp.y and thumb_mcp.y > thumb_ip.y and thumb_ip.y > thumb_tip.y \
        and index_pip.x > thumb_tip.x and index_pip.y < index_dip.y and thumb_tip.y < index_tip.y

    return is_six

def is_seven(hand_landmarks):
    landmarks = hand_landmarks.landmark
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]

    thumb_cmc = landmarks[mp_hands.HandLandmark.THUMB_CMC]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]

    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]
    
    is_seven = equality(middle_pip.y, ring_pip.y) and equality(middle_pip.y, pinky_pip.y) \
        and thumb_cmc.y > thumb_mcp.y and thumb_mcp.y > thumb_ip.y and thumb_ip.y > thumb_tip.y \
        and index_tip.y < index_dip.y and index_dip.y < index_pip.y and index_pip.y < index_mcp.y

    return is_seven

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
                handedness = results.multi_handedness[0].classification[0].label
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS) 
                if handedness == "Right":

                    # Check for gun gesture
                    if is_gun_gesture(hand_landmarks):
                        print("Gun Gesture Recognized")
                        #pyautogui.press('space')

                    elif is_one(hand_landmarks):
                        print("1 Finger Recognized")

                    elif is_two(hand_landmarks):
                        print("2 Fingers Recognized")

                    elif is_three(hand_landmarks):
                        print("3 Finger Recognized")

                    elif is_four(hand_landmarks):
                        print("4 Fingers Recognized")

                    elif is_five(hand_landmarks):
                        print("5 Fingers Recognized")

                    elif is_six(hand_landmarks):
                        print("6 Fingers Recognized")
                
                    elif is_seven(hand_landmarks):
                        print("7 Fingers Recognized")

        # Display image
        cv2.imshow('MediaPipe Hands', image)
        
        # Exit on ESC key
        if cv2.waitKey(5) & 0xFF == 27: 
            break 

cap.release()
cv2.destroyAllWindows()
