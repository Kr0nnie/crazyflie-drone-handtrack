from cvzone.HandTrackingModule import HandDetector
import cv2

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand1 = hands[0]
        lmList1 =hand1["lmList"]
        bbox1 = hand1["bbox"]
        centerpoint1 = hand1["center"]
        handType1 = hand1["type"]

        fingers1 = detector.fingersUp(hand1)
        print(fingers1)
        
        if len(hands)==2:
            hand2 = hands[1]
            lmList2 =hand2["lmList"]
            bbox2 = hand2["bbox"]
            centerpoint2 = hand2["center"]
            handType2 = hand2["type"]

            fingers2 = detector.fingersUp(hand2)

            print(fingers1,fingers2)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)

