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

from cubePoints import *


def DrawLines(img, points):
    for i in range(1, 17):
         x1 = points[0, i - 1]
         y1 = points[1, i - 1]
         x2 = points[0, i]
         y2 = points[1, i]
         cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)
    return img

def getCubePoints(center, size, chessSquare_size):

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

    # bottom
    points.append([center[0] - size, center[1] - size, center[2] - 2 * size])  # (0)5
    points.append([center[0] - size, center[1] + size, center[2] - 2 * size])  # (1)7
    points.append([center[0] + size, center[1] + size, center[2] - 2 * size])  # (2)8
    points.append([center[0] + size, center[1] - size, center[2] - 2 * size])  # (3)6
    points.append([center[0] - size, center[1] - size, center[2] - 2 * size])  # same as first to close plot

    # top
    points.append([center[0] - size, center[1] - size, center[2]])  # (5)1
    points.append([center[0] - size, center[1] + size, center[2]])  # (6)3
    points.append([center[0] + size, center[1] + size, center[2]])  # (7)4
    points.append([center[0] + size, center[1] - size, center[2]])  # (8)2
    points.append([center[0] - size, center[1] - size, center[2]])  # same as first to close plot

    # vertical sides
    points.append([center[0] - size, center[1] - size, center[2]])
    points.append([center[0] - size, center[1] + size, center[2]])
    points.append([center[0] - size, center[1] + size, center[2] - 2 * size])
    points.append([center[0] + size, center[1] + size, center[2] - 2 * size])
    points.append([center[0] + size, center[1] + size, center[2]])
    points.append([center[0] + size, center[1] - size, center[2]])
    points.append([center[0] + size, center[1] - size, center[2] - 2 * size])
    points = dot(points, chessSquare_size)
    return array(points).T


def update(img):
    global cameraCenter
    image = copy(img)


    if Undistorting:  # Use previous stored camera matrix and distortion coefficient to undistort the image
        ''' <004> Here Undistoret the image'''
        image = cv2.undistort(image, cameraMatrix, distCoeffs)

    if (ProcessFrame):

        ''' <005> Here Find the Chess pattern in the current frame'''
        retval, currentCorners = cv2.findChessboardCorners(image, (9, 6))
        patternFound = retval

#         cv2.drawChessboardCorners(image, (9, 6), currentCorners, patternFound)

        if patternFound == True:

            ''' <006> Here Define the cameraMatrix P=K[R|t] of the current frame'''

            if debug:
                # Method 2 (using solvepnp)
                pattern_size = (9, 6)
                pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
                pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
                pattern_points *= chessSquare_size
                obj_points = [pattern_points]
                obj_points.append(pattern_points)
                obj_points = np.array(obj_points, np.float64).T
                obj_points = obj_points[:, :, 0].T

                found, rvecs_new, tvecs_new = GetObjectPos(obj_points, currentCorners, cameraMatrix, distCoeffs)
                rot, jacobian = cv2.Rodrigues(rvecs_new)
                rotationTranslationMatrix = np.hstack((rot, tvecs_new))
                cam = Camera(dot(cameraMatrix, rotationTranslationMatrix))
                cam.factor()

            else:
                # Method 1 (using 2 camera approach)
                # this one is much worse, probably because it computes the homography between two
                # frames, both of which can have distortions (relative to each other, and absolute)
                # Method 2 estimates camera position just from the calibration pattern in the processed image
                idx = np.array([0, 8, -9, -1])
                firstPoints = []
                currentPoints = []

                for i in idx:
                    fp = imagePointsFirst[i]
                    cp = currentCorners[i][0]

                    firstPoints.append(fp)
                    currentPoints.append(cp)

                firstPoints = np.array(firstPoints)
                currentPoints = np.array(currentPoints)

                # between corners in the first captured reference image and the currently captured image
                homography = estimateHomography(firstPoints, currentPoints)

                # Cam
                cam = Camera(dot(homography, originalProjection))

                calibrationInverse = np.linalg.inv(cameraMatrix)
                rot = dot(calibrationInverse, cam.P[:, :4])

                r1, r2, r3, t = tuple(hsplit(rot, 4))
                r3 = cross(r1.T, r2.T).T

                rotationTranslationMatrix = np.hstack((r1, r2, r3, t))

                cam.P = dot(cameraMatrix, rotationTranslationMatrix)
                cam.factor()

            if ShowText:
                ''' <011> Here show the distance between the camera origin and the world origin in the image'''

                camCenter = cam.center()
                distance = sqrt(pow(camCenter[0], 2) + pow(camCenter[1], 2) + pow(camCenter[2], 2))

                cv2.putText(image, str("frame:" + str(frameNumber)), (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))  # Draw the text
                cv2.putText(image, str("dist:" + str(distance)), (20, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))  # Draw the text

            ''' <008> Here Draw the world coordinate system in the image'''

            # Draw Origin
            origin = cam.project(np.array([[0], [0], [0], [1]]))
            cv2.circle(image, getPoint(origin), 5, (255, 255, 0), -1)

            # Draw X Axis
            xunit = cam.project(np.array([[5], [0], [0], [1]]))
            cv2.line(image, getPoint(origin), getPoint(xunit), (255, 0, 0))
            cv2.circle(image, getPoint(xunit), 5, (255, 0, 0), -1)

            # Draw Y Axis
            yunit = cam.project(np.array([[0], [5], [0], [1]]))
            cv2.line(image, getPoint(origin), getPoint(yunit), (255, 0, 0))
            cv2.circle(image, getPoint(yunit), 5, (0, 255, 0), -1)

            # Draw Z Axis
            zunit = cam.project(np.array([[0], [0], [5], [1]]))
            cv2.line(image, getPoint(origin), getPoint(zunit), (255, 0, 0))
            cv2.circle(image, getPoint(zunit), 5, (0, 0, 255), -1)

            if TextureMap:
                ''' <010> Here Do the texture mapping and draw the texture on the faces of the cube'''
                TopFaceCornerNormals, RightFaceCornerNormals, LeftFaceCornerNormals, UpFaceCornerNormals, DownFaceCornerNormals = CalculateFaceCornerNormals(TopFace, RightFace, LeftFace, UpFace, DownFace)

                cornerNormals = {"Top": TopFaceCornerNormals,
                                 "Right": RightFaceCornerNormals,
                                 "Left": LeftFaceCornerNormals,
                                 "Up": UpFaceCornerNormals,
                                 "Down": DownFaceCornerNormals}

                # List all the textures that we want to apply
                textureNames = ["Top", "Right", "Left", "Up", "Down"]

                for texName in textureNames:
                    # textures variable contains all the loaded textures indexed by their name
                    texture = textures[texName]

                    # faces variable contains all the faces indexed by their name
                    # project the face into the current image
                    face = normalizeHomogenious(cam.project(toHomogenious(faces[texName])))
                    face = face[:2, ].T

                    # give the texture some coordinates so it can be transformed too
                    m, n, c = texture.shape
                    texFace = [[0, 0],
                               [0, m],
                               [n, m],
                               [n, 0]]


                    texFace = np.array(texFace, dtype=float64)

                    # transform and render the texture
                    homography, mask = cv2.findHomography(texFace, face)
                    texture = cv2.warpPerspective(texture, homography, (image.shape[1], image.shape[0]))

                    ''' <012> Here Remove the hidden faces'''
                    # Normals
                    # gather the points belonging to the currently evaluated face
                    face2 = faces[texName]
                    point1 = np.array(face2[:, 0])
                    point2 = np.array(face2[:, 1])
                    point3 = np.array(face2[:, 2])
                    point4 = np.array(face2[:, 3])

                    # calculate displacement vectors between the point pairs
                    displacement1 = point1 - point2
                    displacement2 = point2 - point3

                    # calculate normal from displacement vectors as a unitary vector
                    normal = cross(displacement2, displacement1)
                    normal = normal / np.linalg.norm(normal)

                    # calculate points for face center and normal end (used to draw the normals)
                    faceCenter = (point1 + point2 + point3 + point4) / 4
                    normalEnd = faceCenter + normal

                    # project the normal points
                    base = cam.project(toHomogenious(np.reshape(faceCenter, (3, 1))))
                    tip = cam.project(toHomogenious(np.reshape(normalEnd, (3, 1))))

                    cv2.line(image, getPoint(base), getPoint(tip), (255, 255, 0))

                    # get camera center
                    camCenter = cam.center()
                    camCenter = np.reshape(camCenter, (1, 3))
                    camCenter = np.array(camCenter)[0]

                    # write to global variable, used to change light positions
                    # using mouse callback
                    cameraCenter = camCenter

                    # calculate unitary vector of the camera (look at)
                    cameraVector = camCenter - faceCenter
                    cameraVector = cameraVector / np.linalg.norm(cameraVector)

                    # the angle between the face normal and the camera vector
                    angle = arccos(dot(normal, cameraVector))

                    # take care of some edge cases
                    if math.isnan(angle):
                        if cameraVector == normal:
                            angle = 0.0
                        else:
                            angle = np.pi

                    angle = degrees(angle)

                    # if the angle is > 90 degrees, it means that the face is not visible from the camera
                    # i.e facing away from camera, and therefore occluded by other faces, facing towards
                    # the camera
                    if angle > 90.0:
                        continue

                    ''' <013> Here Remove the hidden faces'''
                    # Draw texture
                    mask = np.empty((image.shape[0], image.shape[1], 3), dtype=uint8)
                    mask.fill(255)
                    cv2.fillPoly(mask, np.array([face], 'int32'), (0, 0, 0))

                    masked = cv2.bitwise_and(mask, image)
                    image = cv2.bitwise_or(masked, texture)

                    if Shade:
                        image = ShadeFace(image, face2, cornerNormals[texName], cam)

            if ProjectPattern:
                ''' <007> Here Test the camera matrix of the current view by projecting the pattern points'''
                for corner in currentCorners:
                    corner = corner[0]
                    cv2.circle(image, (int(corner[0]), int(corner[1])), 3, (0, 0, 255), -1)

            if WireFrame:
                ''' <009> Here Project the box into the current camera image and draw the box edges'''
                box2 = cam.project(toHomogenious(box))
                image = DrawLines(image, box2)

    cv2.imshow('Web cam', image)
    cv2.setMouseCallback("Web cam", changeLightSource)
    global result
    result = copy(image)

def getImageSequence(capture, fastForward):
    '''Load the video sequence (fileName) and proceeds, fastForward number of frames.'''
    global frameNumber

    for t in range(fastForward):
        isSequenceOK, originalImage = capture.read()  # Get the first frames
        frameNumber = frameNumber + 1
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
    print 'w: wire frame'
    print 'f: flat/phong shading'
    print 'h: shading'
    print 'x: do something!'


def run(speed):

    '''MAIN Method to load the image sequence and handle user inputs'''

    #--------------------------------video
#     capture = cv2.VideoCapture("Pattern.avi")
#     camera
    capture = cv2.VideoCapture(0)

    image, isSequenceOK = getImageSequence(capture, speed)

    if(isSequenceOK):
        update(image)
        printUsage()

    while(isSequenceOK):
        OriginalImage = copy(image)


        inputKey = cv2.waitKey(1)

        if inputKey == 32:  #  stop by SPACE key
            update(OriginalImage)
            if speed == 0:
                speed = tempSpeed;
            else:
                tempSpeed = speed
                speed = 0;

        if (inputKey == 27) or (inputKey == ord('q')):  #  break by ECS key
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

        if inputKey == ord('f') or inputKey == ord('F'):
            global FlatShade
            if FlatShade:
                FlatShade = False;
            else:
                FlatShade = True;
            update(OriginalImage)

        if inputKey == ord('h') or inputKey == ord('H'):
            global Shade
            if Shade:
                Shade = False;
            else:
                Shade = True;
            update(OriginalImage)

        if inputKey == ord('s') or inputKey == ord('S'):
            name = 'Saved Images/Frame_' + str(frameNumber) + '.png'
            cv2.imwrite(name, result)

        if (speed > 0):
            update(image)
            image, isSequenceOK = getImageSequence(capture, speed)



#---Global variables
global cameraMatrix
global distortionCoefficient
global homographyPoints
global calibrationPoints
global calibrationCamera
global chessSquare_size
global lightPosition
global cameraPosition

ProcessFrame = False
Undistorting = False
WireFrame = False
ShowText = True
TextureMap = True
ProjectPattern = False
Shade = True
FlatShade = False
debug = True

lightPosition = None
cameraPosition = None

frameNumber = 0


chessSquare_size = 2

box = getCubePoints([4, 2.5, 0], 1, chessSquare_size)
box = getCubePoints([4, 2.5, 0], 1, chessSquare_size)

# box = cube_points((0, 0, 0), 0.1)


i = array([ [0, 0, 0, 0], [1, 1, 1, 1] , [2, 2, 2, 2]  ])  # indices for the first dim
j = array([ [0, 3, 2, 1], [0, 3, 2, 1] , [0, 3, 2, 1]  ])  # indices for the second dim
TopFace = box[i, j]


i = array([ [0, 0, 0, 0], [1, 1, 1, 1] , [2, 2, 2, 2]  ])  # indices for the first dim
j = array([ [3, 8, 7, 2], [3, 8, 7, 2] , [3, 8, 7, 2]  ])  # indices for the second dim
RightFace = box[i, j]


i = array([ [0, 0, 0, 0], [1, 1, 1, 1] , [2, 2, 2, 2]  ])  # indices for the first dim
j = array([ [5, 0, 1, 6], [5, 0, 1, 6] , [5, 0, 1, 6]  ])  # indices for the second dim
LeftFace = box[i, j]


i = array([ [0, 0, 0, 0], [1, 1, 1, 1] , [2, 2, 2, 2]  ])  # indices for the first dim
j = array([ [5, 8, 3, 0], [5, 8, 3, 0] , [5, 8, 3, 0] ])  # indices for the second dim
UpFace = box[i, j]


i = array([ [0, 0, 0, 0], [1, 1, 1, 1] , [2, 2, 2, 2]  ])  # indices for the first dim
j = array([ [1, 2, 7, 6], [1, 2, 7, 6], [1, 2, 7, 6] ])  # indices for the second dim
DownFace = box[i, j]

textures = dict()
for tex in ["Top", "Right", "Left", "Down", "Up"]:
    textures[tex] = cv2.imread("Images/" + tex + ".jpg")

faces = {"Top": TopFace,
         "Right": RightFace,
         "Left": LeftFace,
         "Down": DownFace,
         "Up": UpFace}

'''----------------------------------------'''
'''----------------------------------------'''

def getPoint(p):
    return (int(p[0]), int(p[1]))

def ShadeFace(image, points, cornerNormals, cam, drawIntermediate=False):
    '''
    Add shading to the image
    
    Parameters:
        image [array]: the input image 
        points [array]: points belonging to the face being shaded
        cornerNormals [array]: normal vector for each point
        cam [Camera]: the camera object used to project
        drawIntermediate [bool]: wether or not to draw intermediate steps, used for debugging
    
    Returns:
        image [array]: output image with shading applied for the curent face
    '''
    global shadeRes

    # shadeRes is the resolution of the "overlay" texture we add to the
    # already existing texture to simulate shading
    shadeRes = 10

    # get the dimensions of the input image
    videoHeight, videoWidth, vd = np.array(image).shape

    # project the face into the image
    points_projected = cam.project(toHomogenious(points))
    points_projected = np.array([[int(points_projected[0, 0]), int(points_projected[1, 0])],
                                 [int(points_projected[0, 1]), int(points_projected[1, 1])],
                                 [int(points_projected[0, 2]), int(points_projected[1, 2])],
                                 [int(points_projected[0, 3]), int(points_projected[1, 3])]])

    # create an empty image that will be used to store lighting information
    square = np.array([[0, 0], [shadeRes - 1, 0], [shadeRes - 1, shadeRes - 1], [0, shadeRes - 1]])

    # calculate homography to project the lighting texture onto the face in the output image
    homography = estimateHomography(square, points_projected)

    # light texture of dimesions shadeRes x shadeRes with calculated changes to the face texture
    light = CalculateShadeMatrix(image, shadeRes, points, cornerNormals, cam)

    # project light texture onto the final image
    light = cv2.warpPerspective(light, homography, (videoWidth, videoHeight), flags=cv2.INTER_LINEAR)

    # get individual channels from the light texture
    Mr, Mg, Mb = cv2.split(light)

    if drawIntermediate:
        img = cv2.merge((Mr0, Mg0, Mb0))
        cv2.imshow("Face", img)

    # image is BGR
    b, g, r = cv2.split(image)

    # mask for combining light and texture
    whiteMask = np.zeros_like(r)
    cv2.fillConvexPoly(whiteMask, points_projected, (255, 255, 255))

    # multiply texture and light under the mask
    r[nonzero(whiteMask > 0)] = np.clip(r[nonzero(whiteMask > 0)] * Mr[nonzero(whiteMask > 0)], 0, 255)
    g[nonzero(whiteMask > 0)] = np.clip(g[nonzero(whiteMask > 0)] * Mg[nonzero(whiteMask > 0)], 0, 255)
    b[nonzero(whiteMask > 0)] = np.clip(b[nonzero(whiteMask > 0)] * Mb[nonzero(whiteMask > 0)], 0, 255)

    # recombine the image to BGR
    image = cv2.merge((b, g, r))

    return image

def CalculateShadeMatrix(image, shadeRes, points, cornerNormals, cam):
    '''
    Calculate the actual lighting "texture"
    
    Parameters:
        image [array]: image where the texture will be drawn to
        shadeRes [int]: resolution of the lighting texture, since shading is
                        intensive, we want to keep this as small as possible
        points [array]: corners of the face where the lighting is calculated
        cornerNormals [array]: normals at those points
        cam [Camera]: camera used to project the points to the image
    
    Returns:
        (R,G,B)[shadeRes, shadeRes]: lighting "texture" for the face split to RGB components
    '''

    # we store the light position in this global variable from the mouse callback
    global lightPosition

    # create the empty lighting texture
    shade = np.zeros((shadeRes, shadeRes, 3))

    # Ambient Light
    IA = np.array([5.0, 5.0, 5.0])

    # Point Light
    IP = np.array([5.0, 5.0, 5.0])

    # attenuation
    fatt = 1

    # Material
    ka = np.array([0.2, 0.2, 0.2])  # ambient
    kd = np.array([0.3, 0.3, 0.3])  # diffuse
    ks = np.array([0.7, 0.7, 0.7])  # specular

    # used for specular
    alpha = 100

    # Normal
    point1 = np.array(points[:, 0])
    point2 = np.array(points[:, 1])
    point3 = np.array(points[:, 2])
    point4 = np.array(points[:, 3])

    faceNormal = GetFaceNormal(points)

    faceCenter = (point1 + point2 + point3 + point4) / 4

    # camera center
    camCenter = cam.center()
    camCenter = np.reshape(camCenter, (1, 3))
    camCenter = np.array(camCenter)[0]

    # read light position from global (mouse callback) if set
    if lightPosition == None:
        lightPos = camCenter
    else:
        lightPos = lightPosition

    # unitary vector from the camera to the face center
    viewVector = camCenter - faceCenter
    viewVector = viewVector / np.linalg.norm(viewVector)

    # unitary vector from the light source to the face center
    lightIncidenceVector = lightPos - faceCenter
    lightIncidenceVector = lightIncidenceVector / np.linalg.norm(lightIncidenceVector)

    # reflection vector from the face (depends on the face rotation and light position)
    lightReflectionVector = 2 * dot(lightIncidenceVector, faceNormal) * faceNormal - lightIncidenceVector

    # decide if we should draw flat or phong
    if FlatShade:
         # light intensity
        light = max(dot(faceNormal, lightIncidenceVector), 0)

         # specular intensity
        spec = pow(dot(lightReflectionVector, viewVector), alpha)

        # calculate the light (its the same all over the shade texture, hence the "flat")
        shade[:, :, ] = IA * ka + IP * kd * light + IP * ks * spec
    else:
        # Phong shading, now we need to iterate through all of the points in the shade texture
        for y, row in enumerate(shade):
            for x, value in enumerate(row):
                # interpolated normal for this point on the face, takes into account face corner normals
                interpolatedFaceNormal = BilinearInterpo(shadeRes, x, y, cornerNormals, True)
                interpolatedFaceNormal = np.array(interpolatedFaceNormal)

                # light intensity
                light = max(dot(interpolatedFaceNormal, lightIncidenceVector), 0)

                # specular intensity
                lightReflectionVector = 2 * dot(lightIncidenceVector, interpolatedFaceNormal) * interpolatedFaceNormal - lightIncidenceVector
                spec = pow(max(0, dot(lightReflectionVector, viewVector)), alpha)

                # put it all together
                shade[y][x] = IA * ka + IP * kd * light + IP * ks * spec

    # return by channel
#     return shade[:, :, 0], shade[:, :, 1], shade[:, :, 2]
    return shade


def changeLightSource(event, x, y, flag, param):
    if event != 1:
        return

    global lightPosition, cameraCenter

    cx = 320
    cy = 240

    x = 100.0 * ((x - cx) / 320.0)
    y = 100.0 * ((y - cy) / 240.0)

    lightPosition = np.array([x, y, cameraCenter[2]])



''' <000> Here Call the cameraCalibrate2 from the SIGBTools to calibrate the camera and saving the data'''
# RecordVideoFromCamera()
# cameraCalibrate2()

''' <001> Here Load the numpy data files saved by the cameraCalibrate2'''
cameraMatrix = np.load("numpyData/camera_matrix.npy")
distCoeffs = np.load("numpyData/distortionCoefficient.npy")
rvec = np.load("numpyData/rotatioVectors.npy")
tvec = np.load("numpyData/translationVectors.npy")

imagePointsFirst = np.load("numpyData/img_points_first.npy")
objectPoints = np.load("numpyData/obj_points.npy")
#
# print(cameraMatrix)
# print(distCoeffs)

''' <002> Here Define the camera matrix of the first view image (01.png) recorded by the cameraCalibrate2'''

rotation, jacobian = cv2.Rodrigues(rvec[0])
translation = tvec[0]
#
originalProjection = dot(cameraMatrix, hstack((rotation, translation)))

''' <003> Here Load the first view image (01.png) and find the chess pattern and store the 4 corners of the pattern needed for homography estimation'''
image = cv2.imread("01.png")
retval, originalCorners = cv2.findChessboardCorners(image, (9, 6))

# for point in objectPoints[0]:
#    point = np.append(point, 1).T
#    point = dot(originalProjection, point).T
#    if point[2] != 0:
#        point[0] = point[0] / point[2]
#        point[1] = point[1] / point[2]
#
#    cv2.circle(image, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)
#
# cv2.imshow("Test", image)
# cv2.waitKey(0)

run(1)
