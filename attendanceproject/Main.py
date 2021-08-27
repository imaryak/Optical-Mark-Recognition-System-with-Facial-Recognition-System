import cv2
import numpy as np
import utlis


########################################################################
webCamFeed = True
heightImg = 700
widthImg  = 700
pathImage = "Img2/1.jpeg"
questions=5
choices=5
ans=[1,2,0,1,4]
webCamFeed=True
cameraNo=1
########################################################################

cap=cv2.VideoCapture(1)
cap.set(10,150)

while True:
    if webCamFeed:success,img=cap.read()
    else:img=cv2.imread(pathImage)

    #PREPROCESSING
    img=cv2.imread(pathImage)
    img=cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
    imgContours = img.copy()
    imgFinal = img.copy()
    imgBiggestContours = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
    imgCanny = cv2.Canny(imgBlur,10,70) # APPLY CANNY TO DETECT EDGES

    try:
        #FINDING ALL CONTOURS
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # FIND ALL CONTOURS
        cv2.drawContours(imgContours,contours,-1,(0,255,0),10)
        #FIND RECTANGLES
        rectCon=utlis.rectContour(contours)
        biggestContour=utlis.getCornerPoints(rectCon[0])

        gradePoints=utlis.getCornerPoints(rectCon[1])
        print(biggestContour)

        if biggestContour.size !=0 and gradePoints.size !=0:
            cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),20)
            cv2.drawContours(imgBiggestContours,gradePoints, -1, (255, 0, 0), 20)

            biggestContour=utlis.reorder(biggestContour)
            gradePoints= utlis.reorder(gradePoints)

            pt1=np.float32(biggestContour)
            pt2=np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
            matrix =cv2.getPerspectiveTransform(pt1,pt2)
            imgWarpColored=cv2.warpPerspective(img,matrix,(widthImg,heightImg))

            ptG1 = np.float32(gradePoints)
            ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
            #cv2.imshow("Grade",imgGradeDisplay)

            #Apply threshold
            imgWarpGray=cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
            imgThresh=cv2.threshold(imgWarpGray,170,255,cv2.THRESH_BINARY_INV)[1]

            boxes=utlis.splitBoxes(imgThresh)
            #cv2.imshow("Test",boxes[2])
            #print(cv2.countNonZero(boxes[1),cv2.countNonZero(boxes[2]))

            #GETTING NO ZERO PIXEL VALUES OF EACH BOX
            myPixelVal=np.zeros((questions,choices))
            countC=0
            countR=0
            for image in boxes:
                totalPixels=cv2.countNonZero(image)
                myPixelVal[countR][countC]=totalPixels
                countC+=1
                if(countC==choices):countR+=1;countC=0
            #print(myPixelVal)

            #FINDING INDEX VALUES OF THE MARKINGS
            myIndex=[]
            for x in range(0,questions):
                arr=myPixelVal[x]
                #print("arr",arr)
                myIndexVal=np.where(arr==np.max(arr))
                #print(myIndexVal[0])
                myIndex.append(myIndexVal[0][0])
            #print(myIndex)

            #GRADING
            grading=[]
            for x in range(0,questions):
                if ans[x]==myIndex[x]:
                    grading.append(1)
                else:grading.append(0)
            #print(grading)
            score=(sum(grading)/questions)*100#FINAL GRADE
            print(score)

            #DISPLAYING ANSWERS
            imgResult=imgWarpColored.copy()
            imgResult=utlis.showAnswers(imgResult, myIndex, grading, ans, questions, choices)
            imRawDrawing=np.zeros_like(imgWarpColored)
            imRawDrawing = utlis.showAnswers(imRawDrawing ,myIndex, grading, ans, questions, choices)
            invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
            imgInvWarp = cv2.warpPerspective(imRawDrawing, invMatrix, (widthImg, heightImg))

            imgRawGrade=np.zeros_like(imgGradeDisplay)
            cv2.putText(imgRawGrade,str(int(score))+"%",(60,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3)
            invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))

            imgFinal=cv2.addWeighted(imgFinal,1,imgInvWarp,1,0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)



        imgBlank=np.zeros_like(img)
        imgArray=([img,imgGray,imgBlur,imgCanny],
                  [imgContours,imgBiggestContours,imgWarpColored,imgThresh],
                  [imgResult,imRawDrawing,imgInvWarp,imgFinal])
    except:
        imgBlank=np.zeros_like(img)
        imgArray=([img,imgGray,imgBlur,imgCanny],
                  [imgBlank,imgBlank,imgBlank,imgBlank],
                  [imgBlank,imgBlank,imgBlank,imgBlank])


    lables=[["Original","Gray","Blur","Canny"],
            ["Contours","Biggest Con","Warp","Threshold"],
            ["Result","Raw Drawing","Inv Warp","Final"]]

    imgStacked=utlis.stackImages(imgArray,0.4)

    cv2.imshow("Final Result",imgFinal)
    cv2.imshow("Stacked Images",imgStacked)
    if cv2.waitKey(1)& 0xFF ==ord('$'):
        cv2.imwrite("FinalResult.jpg",imgFinal)
        cv2.waitkey(300)