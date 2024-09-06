import cv2

# load model pendeteksi objek
model_motor = cv2.dnn.readNetFromDarknet('download.cfg', 'motor.weights')
model_manusia = cv2.dnn.readNetFromDarknet('download.cfg', 'manusia.weights')

# load nama kelas
with open('motor.names', 'rt') as f:
    motor_labels = f.read().rstrip('\n').split('\n')

with open('manusia.names', 'rt') as f:
    manusia_labels = f.read().rstrip('\n').split('\n')

# konfigurasi video stream
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set lebar video stream
cap.set(4, 480)  # set tinggi video stream

while True:
    # baca setiap frame dari video stream
    ret, frame = cap.read()

    # lakukan deteksi objek pada setiap frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)

    model_motor.setInput(blob)
    motor_detections = model_motor.forward()

    model_manusia.setInput(blob)
    manusia_detections = model_manusia.forward()

    # tampilkan hasil deteksi objek pada setiap frame
    for detection in motor_detections:
        scores = detection[5:]
        class_id = scores.argmax()
        confidence = scores[class_id]
        if confidence > 0.5 and class_id == 0:
            center_x = int(detection[0] * frame.shape[1])
            center_y = int(detection[1] * frame.shape[0])
            width = int(detection[2] * frame.shape[1])
            height = int(detection[3] * frame.shape[0])
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 0, 255), 2)
            cv2.putText(frame, motor_labels[class_id], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for detection in manusia_detections:
        scores = detection[5:]
        class_id = scores.argmax()
        confidence = scores[class_id]
        if confidence > 0.5 and class_id == 0:
            center_x = int(detection[0] * frame.shape[1])
            center_y = int(detection[1] * frame.shape[0])
            width = int(detection[2] * frame.shape[1])
            height = int(detection[3] * frame.shape[0])
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
            cv2.putText(frame, manusia_labels[class_id], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)

    # tampilkan frame yang telah diolah
    cv2.imshow('Motor dan Manusia Detector', frame)

    # keluar dari program jika user menekan tombol 'q
    if cv2.waitKey(1) == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()
