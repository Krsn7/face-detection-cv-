import cv2

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check if the cascade is loaded successfully
if face_cascade.empty():
    print("Error: Could not load cascade file. Check the file path.")
    exit()

# Initialize the webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, img = webcam.read()
    if not ret:
        print("Error: Could not read frame. Exiting...")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Display the frame with detected faces
    cv2.imshow("Face Detection", img)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("Exiting... 'q' key pressed.")
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
