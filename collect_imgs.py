import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 100

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    done = False
    while not done:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        
        if frame.shape[0] > 0 and frame.shape[1] > 0:
            cv2.imshow('frame', frame)
        else:
            print("Error: Invalid frame size.")
            break

        key = cv2.waitKey(25)
        if key == ord('q'):
            done = True

    if not done:
        break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        if frame.shape[0] > 0 and frame.shape[1] > 0:
            cv2.imshow('frame', frame)
        else:
            print("Error: Invalid frame size.")
            break

        key = cv2.waitKey(25)
        if key == ord('q'):
            done = True
            break

        filename = os.path.join(class_dir, '{}.jpg'.format(counter))
        cv2.imwrite(filename, frame)
        counter += 1

    if not done:
        break

cap.release()
cv2.destroyAllWindows()
