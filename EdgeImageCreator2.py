import numpy as np, cv2 as cv
import pickle
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
from fast_histogram import histogram2d
import os
from os import listdir

import matplotlib.pyplot as plt

with open('SSM_Data', 'rb') as f:
    SSM_Data = pickle.load(f)

triangulation = {0: SSM_Data['femTriangulation'],
                 1: SSM_Data['tibTriangulation'],
                 2: SSM_Data['patTriangulation']}

scene = {}
fieldSize = scene['fieldSize'] = 9 * 25.4  # 9 inches
src_to_screen = scene['src_to_screen'] = 950  # on x -axis
src_to_model = scene['src_to_model'] = 850
totalPixels = 1024  # along any direction
origin = -0.5 * np.array([0, fieldSize, fieldSize])
totalPoses = 32
aspectRatio = 1.25

from functions import *

batchSize = 500
alpha = np.linspace(-10, 190, totalPoses)
batchParams = RandomParamGenerator(alpha, src_to_model, batchSize)
X = ModelGenerator(SSM_Data, batchParams["shapeParams"])
XYZModel, CS = ConvertModeltoXYZ_Coord(X)
allVertices, allFaces = CreateTriangleMesh(XYZModel, triangulation)

mesh = o3d.geometry.TriangleMesh()
mesh.triangles = o3d.utility.Vector3iVector(allFaces.astype(np.int32))

yedges, zedges = np.linspace(0, 255, 256), np.linspace(0, 255, 256)
hist = np.zeros((totalPoses, 256, 256))

for i, vertices in enumerate(allVertices):
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    ptCld = mesh.sample_points_uniformly(vertices.shape[0] * 700)
    denseVertices = np.asarray(ptCld.points)

    # TODO  Check this
    translations = batchParams['poseParams'][i, :, 3:6]. \
        reshape(totalPoses, 1, 3)

    # Transformation matrix
    rot_angles = batchParams['poseParams'][i, :, 0:3]
    T = R.from_euler('zxy', rot_angles, degrees=True).as_matrix()

    # Vertices of a particular model in different poses
    poseModels = np.matmul(denseVertices, np.einsum('ijk->ikj', T)) + translations
    # DisplayPointModel(poseModels[0], scene)

    # Intermediate step
    A = poseModels * src_to_screen / (poseModels[:, :, 0].reshape(totalPoses, -1, 1))

    # Find the screen Coordinates when the model is projected on to the screen
    screenCoord = (A - origin) * totalPixels / fieldSize
    screenCoord = screenCoord[:, :, [1, 2]]  # Remove x-coordinates
    del A, poseModels  # Saves memory

    # Create a appropriate bounding box around the knee. Same box should fit all the poses

    widthofKnee = np.zeros([screenCoord.shape[0], 1])
    for pose in range(screenCoord.shape[0]):
        widthofKnee[pose] = np.ptp(screenCoord[pose, :, 0])
    maxWidthofKnee = np.max(widthofKnee)

    boxWidth = maxWidthofKnee + np.random.randint(10, 50)  # boxWidth is the maxWidthofKnee + a random number
    boxHeight = boxWidth * aspectRatio

    # Reduce the rectangle box to 256x256 image
    #                OR
    # Scale down the points such that the imaginary box fits into 256x256 image
    # This way the pressure on the 2D histogram function can be reduced and make it run faster

    xScale = 256 / boxWidth
    yScale = 256 / boxHeight

    screenCoord[:, :, 0] = screenCoord[:, :, 0] * xScale
    screenCoord[:, :, 1] = screenCoord[:, :, 1] * yScale

    for pose in range(screenCoord.shape[0]):
        kneeWidth = np.ptp(screenCoord[pose, :, 0])
        xOffset = np.min(screenCoord[pose, :, 0]) - np.random.randint(0, 256 - kneeWidth)  # 256-kneeWidth must be >0

        kneeHeight = np.ptp(screenCoord[pose, :, 1])
        yOffset = np.random.randint(0, np.abs(256 - kneeHeight) + 1) + np.min(screenCoord[pose, :, 1])

        screenCoord[pose, :, :] = screenCoord[pose, :, :] - [xOffset, yOffset]

    # Display2DPoints(screenCoord[1])
    # Find 2d histogram
    for j in range(totalPoses):
        hist[j, :, :] = np.flip(histogram2d(screenCoord[j, :, 1], screenCoord[j, :, 0], range=[[0, 255], [0, 255]],
                                            bins=[256, 256]), axis=0)
    del screenCoord

    # Arange all the poses in tiles and in the last column encode shape parameters
    tiledImage = TileImages(hist, tiles=16, shapeParams=batchParams['shapeParams'][i])
    path = 'D:\OneDrive - University of Waterloo\Thesis\Projects\IC\Synthetic Images\Histograms'
    for p in range(tiledImage.shape[0]):
        cv.imwrite(os.path.join(path, '%d.png' % (len(listdir(path)) + 1)), np.uint16(tiledImage[p]))

##
# A1 = tiledImage[1]  # ndimage.rotate(hist[15], 180)
# from mayavi import mlab
#
# mlab.imshow(A1, colormap='gray')
# mlab.view(-90, 0)
