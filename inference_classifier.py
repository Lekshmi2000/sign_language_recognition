import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17: 'R', 18: 'S', 19: 'T', 20:'U', 21: 'V', 22: 'W', 23: 'X', 24:'Y', 25: 'Z' }
final_char=""
# Create an image with white background
img = 255 * np.ones((500, 600, 3), dtype=np.uint8)
sentence1="PRESS s TO SAVE"
sentence2="PRESS q TO QUIT"
sentence3="PRESS d TO DELETE"
sentence4="PRESS p TO DISPLAY"
sentence5="The predicted word is : "
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]
        
        key=cv2.waitKey(1)
        if key == 115:
            final_char=final_char+predicted_character
            #print(final_char)    
        

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
    key=cv2.waitKey(1)    
    if key == ord('q'):
       cap.release()
       cv2.destroyAllWindows()
       break
    font = cv2.FONT_HERSHEY_SIMPLEX
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1

    # Position the text at the top right corner with a margin of 10 pixels
    x = 45
    y = 40
    org = (x, y)

# Display the text on the image
    color = (0, 0, 0)
    cv2.putText(img, sentence1, org, font, font_scale, color, thickness, cv2.LINE_AA)
    x=x+300
    org = (x, y)
    cv2.putText(img, sentence2, org, font, font_scale, color, thickness, cv2.LINE_AA)
    y=y+30
    x=45
    org = (x, y)
    cv2.putText(img, sentence3, org, font, font_scale, color, thickness, cv2.LINE_AA)
    x=x+300
    org = (x, y)
    cv2.putText(img, sentence4, org, font, font_scale, color, thickness, cv2.LINE_AA)
    x=70
    y=120
    font_scale=0.8
    org = (x, y)
    cv2.putText(img, sentence5, org, font, font_scale, color, thickness, cv2.LINE_AA)
    x=120
    y=150
    org = (x, y)
    color = (178,34,34)
    font_scale=0.8
    thickness = 2
    key=cv2.waitKey(1)
    if key == 112:   
        cv2.putText(img, final_char, org, font, font_scale, color, thickness, cv2.LINE_AA)
        print(final_char)         
    cv2.imshow('frame', frame)
    cv2.imshow('image', img)
    
