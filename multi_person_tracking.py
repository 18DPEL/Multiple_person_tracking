import cv2

trdict = {'csrt': cv2.TrackerCSRT_create,
          'kcf': cv2.TrackerKCF_create,
          'mil': cv2.TrackerMIL_create}

# Create trackers for five persons
tracker_person1 = trdict['csrt']()
tracker_person2 = trdict['csrt']()
tracker_person3 = trdict['kcf']()
tracker_person4 = trdict['csrt']()
tracker_person5 = trdict['csrt']()

v = cv2.VideoCapture('move.mp4')

ret, frame = v.read()
if not ret or frame is None:
    print("Error: Unable to read the video frame.")
    exit(1)

cv2.imshow('Frame', frame)

# Flags to track if persons are being tracked
tracking_person1 = False
tracking_person2 = False
tracking_person3=  False
tracking_person4=  False
tracking_person5=  False

# Mouse callback function
def on_mouse(event, x, y, flags, param):
    global tracking_person1, tracking_person2, tracking_person3, tracking_person4, tracking_person5
    if event == cv2.EVENT_LBUTTONDOWN:
        if not tracking_person1:
            # Initialize tracker for person 1
            bb_person1 = (x, y, 70, 150)  # You can adjust the size of the bounding box
            tracker_person1.init(frame, bb_person1)
            tracking_person1 = True
        elif not tracking_person2:
            # Initialize tracker for person 2
            bb_person2 = (x, y, 70, 150)  # You can adjust the size of the bounding box
            tracker_person2.init(frame, bb_person2)
            tracking_person2 = True
        elif not tracking_person3:
            # Initialize tracker for person 3
            bb_person3 = (x, y, 70, 150)  # You can adjust the size of the bounding box
            tracker_person3.init(frame, bb_person3)
            tracking_person3 = True
        elif not tracking_person4:
            # Initialize tracker for person 4
            bb_person4 = (x, y, 70, 150)  # You can adjust the size of the bounding box
            tracker_person4.init(frame, bb_person4)
            tracking_person4 = True
        elif not tracking_person5:
            # Initialize tracker for person 5
            bb_person5 = (x, y, 70, 150)  # You can adjust the size of the bounding box
            tracker_person5.init(frame, bb_person5)
            tracking_person5 = True


# Set the mouse callback function
cv2.setMouseCallback('Frame', on_mouse)

while True:
    ret, frame = v.read()
    if not ret or frame is None:
        break

    # Update trackers for each person if tracking is enabled
    if tracking_person1:
        success_person1, box_person1 = tracker_person1.update(frame)
        if success_person1:
            x1, y1, w1, h1 = [int(a) for a in box_person1]
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
        else:
            # If person1 is not tracked, clear the tracker and disable tracking
            tracker_person1.clear()
            tracking_person1 = False

    if tracking_person2:
        success_person2, box_person2 = tracker_person2.update(frame)
        if success_person2:
            x2, y2, w2, h2 = [int(a) for a in box_person2]
            cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
        else:
            # If person2 is not tracked, clear the tracker and disable tracking
            tracker_person2.clear()
            tracking_person2 = False

    if tracking_person3:
        success_person3, box_person3 = tracker_person3.update(frame)
        if success_person3:
            x3, y3, w3, h3 = [int(a) for a in box_person2]
            cv2.rectangle(frame, (x3, y3), (x3 + w3, y3 + h3), (255, 255, 0), 2)
        else:
            # If person3 is not tracked, clear the tracker and disable tracking
            tracker_person3.clear()
            tracking_person3 = False

    if tracking_person4:
        success_person4, box_person4 = tracker_person4.update(frame)
        if success_person4:
            x4, y4, w4, h4 = [int(a) for a in box_person4]
            cv2.rectangle(frame, (x4, y4), (x4 + w4, y4 + h4), (0, 255,0), 2)
        else:
            # If person4 is not tracked, clear the tracker and disable tracking
            tracker_person4.clear()
            tracking_person4 = False

    if tracking_person5:
        success_person5, box_person5 = tracker_person5.update(frame)
        if success_person5:
            x5, y5, w5, h5 = [int(a) for a in box_person5]
            cv2.rectangle(frame, (x5, y5), (x5 + w5, y5 + h5), (255, 0,255), 2)
        else:
            # If person5 is not tracked, clear the tracker and disable tracking
            tracker_person5.clear()
            tracking_person5 = False

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

v.release()
cv2.destroyAllWindows()
