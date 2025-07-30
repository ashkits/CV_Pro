import cv2

# Video source
video_path = r'C:\CV_Pro\PedestrianCounter-master\3552510-hd_1920_1080_30fps.mp4'  # Change this to your video path
cap = cv2.VideoCapture(video_path)

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Boundary for counting
mid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
boundary_line = mid_height - 100

people_in = 0
people_out = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area to consider as a person
            x, y, w, h = cv2.boundingRect(contour)
            center_y = y + h // 2

            # Draw rectangle around detected person
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Check if the person crosses the boundary line
            if center_y < boundary_line and (y + h) > boundary_line:
                people_in += 1
            elif center_y > boundary_line and y < boundary_line:
                people_out += 1

    # Draw the boundary line
    cv2.line(frame, (0, boundary_line), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), boundary_line), (255, 0, 0), 2)

    # Display counts on the frame
    cv2.putText(frame, f'In: {people_in}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Out: {people_out}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Pedestrian Counter', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()