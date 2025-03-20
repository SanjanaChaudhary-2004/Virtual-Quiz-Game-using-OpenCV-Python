import cv2
import csv
import cvzone
import time
from cvzone.HandTrackingModule import HandDetector
cap = cv2.VideoCapture(0)   #capture video from webcam
cap.set(3, 1280)  #width
cap.set(4, 720)   #height
detector = HandDetector(detectionCon=0.8)   #hand detector object

class MCQ(): #class to store mcq data
    def __init__(self, data):
        self.question = data[0] 
        self.choice1 = data[1]
        self.choice2 = data[2]
        self.choice3 = data[3]
        self.choice4 = data[4]
        self.answer = int(data[5])

        self.userAns = None
    
    def update(self,cursor,bboxs):    #created this method to check whether the options were clicked or not

        for x, bbox in enumerate(bboxs): #loop through all the options
            x1,y1,x2,y2 = bbox #bounding box coordinates
            if x1<cursor[0]<x2 and y1<cursor[1]<y2:##check if the cursor is inside the bounding box
                self.userAns = x + 1  #store the option clicked
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),cv2.FILLED) #highlight the option clicked

#Import csv file data
pathCSV = 'mcqs.csv'
with open(pathCSV, newline='\n') as f: 
    reader = csv.reader(f)
    dataAll = list(reader)[1:]

# create object for each mcq
mcqList =[]
for q in dataAll:
    mcqList.append(MCQ(q))

print(len(mcqList))


qNo = 0
qTotal = len(dataAll)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) #image flipped in horizontal direaction
    hands, img = detector.findHands(img, flipType=False) #find hands
    
    if qNo < qTotal:
        mcq = mcqList[qNo]

        img, bbox =cvzone.putTextRect(img,mcq.question,[100,100],2,2,offset=50,border=5)
        img, bbox1 =cvzone.putTextRect(img,mcq.choice1,[100,250],2,2,offset=50,border=5)
        img, bbox2 =cvzone.putTextRect(img,mcq.choice2,[400,250],2,2,offset=50,border=5)
        img, bbox3 =cvzone.putTextRect(img,mcq.choice3,[100,400],2,2,offset=50,border=5)
        img, bbox4 =cvzone.putTextRect(img,mcq.choice4,[400,400],2,2,offset=50,border=5)
    
    
        if hands:
            lmList = hands[0]['lmList'] #list of 21 landmarks
            cursor = lmList[8] #index finger tip landmark
            length, info, img = detector.findDistance(lmList[8][:2], lmList[12][:2], img)#distance between index and middle finger
            if length < 60:
                mcq.update(cursor,[bbox1,bbox2,bbox3,bbox4])
                print(mcq.userAns)
                if mcq.userAns is not None:
                    time.sleep(0.3) #delay to avoid multiple clicks
                    qNo += 1 
    else:
        score = 0
        for mcq in mcqList:
            if mcq.answer == mcq.userAns:
                score += 1
        score = round(score/qTotal*100,2)
        img, _ =cvzone.putTextRect(img,"Quiz Completed",[250,300],2,2,offset=50,border=5 )
        img, _ =cvzone.putTextRect(img,f'Your Score:{score}%',[700,300],2,2,offset=50,border=5 )
    
    # Draw Progress bar
    barValue = 150 + (950 // qTotal) * qNo
    cv2.rectangle(img, (150, 600), (barValue, 650), (0, 255, 0), cv2.FILLED)
    cv2.rectangle(img, (150, 600), (1100, 650), (255, 0, 255), 5)
    img, _ = cvzone.putTextRect(img, f'{round((qNo / qTotal) * 100)}%', [1130, 635], 2, 2, offset=16)

    
    cv2.imshow("Img", img)
    cv2.waitKey(1)
  

  