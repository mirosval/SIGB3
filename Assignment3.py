'''
Created on Apr 11, 2013

@author: Diako Mardanbegi (dima@itu.dk)
'''
from numpy import *
import numpy as np
from pylab import *
from scipy import linalg
import cv2
import cv2.cv as cv
from SIGBTools import *


def DrawLines(img, points):
    for i in range(1, 17):                
         x1 = points[0, i - 1]
         y1 = points[1, i - 1]
         x2 = points[0, i]
         y2 = points[1, i]
         cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0),5) 
    return img

def getCubePoints(center, size,chessSquare_size):

    """ Creates a list of points for plotting
    a cube with plot. (the first 5 points are
    the bottom square, some sides repeated). """    
    points = []
    
    """
    1    2
        5    6
    3    4
        7    8 (bottom)
    
    """

    #bottom
    points.append([center[0]-size, center[1]-size, center[2]-2*size])#(0)5
    points.append([center[0]-size, center[1]+size, center[2]-2*size])#(1)7
    points.append([center[0]+size, center[1]+size, center[2]-2*size])#(2)8
    points.append([center[0]+size, center[1]-size, center[2]-2*size])#(3)6
    points.append([center[0]-size, center[1]-size, center[2]-2*size]) #same as first to close plot
    
    #top
    points.append([center[0]-size,center[1]-size,center[2]])#(5)1
    points.append([center[0]-size,center[1]+size,center[2]])#(6)3
    points.append([center[0]+size,center[1]+size,center[2]])#(7)4
    points.append([center[0]+size,center[1]-size,center[2]])#(8)2
    points.append([center[0]-size,center[1]-size,center[2]]) #same as first to close plot
    
    #vertical sides
    points.append([center[0]-size,center[1]-size,center[2]])
    points.append([center[0]-size,center[1]+size,center[2]])
    points.append([center[0]-size,center[1]+size,center[2]-2*size])
    points.append([center[0]+size,center[1]+size,center[2]-2*size])
    points.append([center[0]+size,center[1]+size,center[2]])
    points.append([center[0]+size,center[1]-size,center[2]])
    points.append([center[0]+size,center[1]-size,center[2]-2*size])
    points=dot(points,chessSquare_size)
    return array(points).T


def update(img):
    image=copy(img)


    if Undistorting:  #Use previous stored camera matrix and distortion coefficient to undistort the image
        ''' <004> Here Undistoret the image''' 
    
    if (ProcessFrame): 

        ''' <005> Here Find the Chess pattern in the current frame'''
        patternFound=True          
    
        if patternFound ==True:        
            
            ''' <006> Here Define the cameraMatrix P=K[R|t] of the current frame'''

        
            if ShowText:
                ''' <011> Here show the distance between the camera origin and the world origin in the image'''                    
                
                cv2.putText(image,str("frame:" + str(frameNumber)), (20,10),cv2.FONT_HERSHEY_PLAIN,1, (255, 255, 255))#Draw the text

            ''' <008> Here Draw the world coordinate system in the image'''            
            
            if TextureMap:
                ''' <010> Here Do the texture mapping and draw the texture on the faces of the cube'''  
                ''' <012> Here Remove the hidden faces'''  

                ''' <013> Here Remove the hidden faces'''  


            if ProjectPattern:                  
                ''' <007> Here Test the camera matrix of the current view by projecting the pattern points''' 
            
        

            if WireFrame:                      
                ''' <009> Here Project the box into the current camera image and draw the box edges''' 
    

    cv2.imshow('Web cam', image)  
    global result
    result=copy(image)

def getImageSequence(capture, fastForward):
    '''Load the video sequence (fileName) and proceeds, fastForward number of frames.'''
    global frameNumber
   
    for t in range(fastForward):
        isSequenceOK, originalImage = capture.read()  # Get the first frames
        frameNumber = frameNumber+1
    return originalImage, isSequenceOK


def printUsage():
    print "Q or ESC: Stop"
    print "SPACE: Pause"     
    print "p: turning the processing on/off "  
    print 'u: undistorting the image'
    print 'i: show info'
    print 't: texture map'
    print 'g: project the pattern using the camera matrix (test)'
    print 's: save frame'
    print 'x: do something!'
    
   
def run(speed): 
    
    '''MAIN Method to load the image sequence and handle user inputs'''   

    #--------------------------------video
    #capture = cv2.VideoCapture("Pattern.avi")
    #--------------------------------camera
    capture = cv2.VideoCapture(0)

    image, isSequenceOK = getImageSequence(capture,speed)       

    if(isSequenceOK):
        update(image)
        printUsage()

    while(isSequenceOK):
        OriginalImage=copy(image)
     
        
        inputKey = cv2.waitKey(1)
        
        if inputKey == 32:#  stop by SPACE key
            update(OriginalImage)
            if speed==0:     
                speed = tempSpeed;
            else:
                tempSpeed=speed
                speed = 0;                    
            
        if (inputKey == 27) or (inputKey == ord('q')):#  break by ECS key
            break    
                
        if inputKey == ord('p') or inputKey == ord('P'):
            global ProcessFrame
            if ProcessFrame:     
                ProcessFrame = False;
                
            else:
                ProcessFrame = True;
            update(OriginalImage)
            
        if inputKey == ord('u') or inputKey == ord('U'):
            global Undistorting
            if Undistorting:     
                Undistorting = False;
            else:
                Undistorting = True;
            update(OriginalImage)     
        if inputKey == ord('w') or inputKey == ord('W'):
            global WireFrame
            if WireFrame:     
                WireFrame = False;
                
            else:
                WireFrame = True;
            update(OriginalImage)

        if inputKey == ord('i') or inputKey == ord('I'):
            global ShowText
            if ShowText:     
                ShowText = False;
                
            else:
                ShowText = True;
            update(OriginalImage)
            
        if inputKey == ord('t') or inputKey == ord('T'):
            global TextureMap
            if TextureMap:     
                TextureMap = False;
                
            else:
                TextureMap = True;
            update(OriginalImage)
            
        if inputKey == ord('g') or inputKey == ord('G'):
            global ProjectPattern
            if ProjectPattern:     
                ProjectPattern = False;
                
            else:
                ProjectPattern = True;
            update(OriginalImage)   
             
        if inputKey == ord('x') or inputKey == ord('X'):
            global debug
            if debug:     
                debug = False;                
            else:
                debug = True;
            update(OriginalImage)   
            
                
        if inputKey == ord('s') or inputKey == ord('S'):
            name='Saved Images/Frame_' + str(frameNumber)+'.png' 
            cv2.imwrite(name,result)
           
        if (speed>0):
            update(image)
            image, isSequenceOK = getImageSequence(capture,speed)          



#---Global variables
global cameraMatrix    
global distortionCoefficient
global homographyPoints
global calibrationPoints
global calibrationCamera
global chessSquare_size
    
ProcessFrame=False
Undistorting=False   
WireFrame=False
ShowText=True
TextureMap=True
ProjectPattern=False
debug=True

frameNumber=0


chessSquare_size=2
            
box = getCubePoints([4, 2.5, 0], 1,chessSquare_size)            


i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [0,3,2,1],[0,3,2,1] ,[0,3,2,1]  ])  # indices for the second dim            
TopFace = box[i,j]


i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [3,8,7,2],[3,8,7,2] ,[3,8,7,2]  ])  # indices for the second dim            
RightFace = box[i,j]


i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [5,0,1,6],[5,0,1,6] ,[5,0,1,6]  ])  # indices for the second dim            
LeftFace = box[i,j]


i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [5,8,3,0], [5,8,3,0] , [5,8,3,0] ])  # indices for the second dim            
UpFace = box[i,j]


i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [1,2,7,6], [1,2,7,6], [1,2,7,6] ])  # indices for the second dim            
DownFace = box[i,j]



'''----------------------------------------'''
'''----------------------------------------'''



''' <000> Here Call the cameraCalibrate2 from the SIGBTools to calibrate the camera and saving the data''' 
''' <001> Here Load the numpy data files saved by the cameraCalibrate2''' 
''' <002> Here Define the camera matrix of the first view image (01.png) recorded by the cameraCalibrate2''' 
''' <003> Here Load the first view image (01.png) and find the chess pattern and store the 4 corners of the pattern needed for homography estimation''' 



run(1)