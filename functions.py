from numpy.random import rand, normal
import numpy as np, cv2 as cv


def RandomParamGenerator(alpha, src_to_model, batchSize):
    size = len(alpha)

    # Pose Parameters (alpha,beta,gamma,y,z)
    # alpha- Rotation along z-axis
    # beta- Rotation along y-axis
    # gamma- Rotation along x-axis

    # alpha = alpha + 2 * deltaAlpha * (0.5 - rand(size, 1))
    # beta = 2 * 0 * (0.5 - rand(size, 1))
    # gamma = 2 * 0 * (0.5 - rand(size, 1))
    # x = 2 * 0 * (0.5 - rand(size, 1))
    # y = 2 * 0 * (0.5 - rand(size, 1))
    # z = 2 * 0 * (0.5 - rand(size, 1))

    delta = [0, 0, 0, 0, 0, 0]  # delta of [alpha,beta, gamma,x,y,z]
    poseParams = 2 * (0.5 - normal(0, 1, (batchSize, size, 6)))
    poseParams = poseParams * delta

    poseParams[:, :, 0] = poseParams[:, :, 0] + alpha
    poseParams[:, :, 3] = src_to_model  # src_to_model = 300 + 2 * 0 * (0.5 - rand(size, 1))  # on x-axis
    #
    # Shape parameters- 49 values
    delta = 3
    shapeParams = 2 * delta * (0.5 - rand(batchSize, 49))

    Allparams = {'poseParams': poseParams, 'shapeParams': shapeParams}
    return Allparams


def ModelGenerator(SSM_Data, shapeParams):
    # means and std deviations for each PC
    mu_pcs = SSM_Data['PC_DATA'][0, :]
    sigma_pcs = SSM_Data['PC_DATA'][1, :]

    NewPCs = mu_pcs + np.multiply(shapeParams, np.transpose(sigma_pcs))

    # Multiply from PC space to Cartesain space
    # Realign data with respect to the new specimen by adding MU
    X = np.matmul(NewPCs, SSM_Data['COEFF']) + SSM_Data['MU']

    return X


def ConvertModeltoXYZ_Coord(X):
    def organize(bone, batchSize):
        length = int(bone.shape[1] / 3)
        XYZcoord = np.ones((batchSize, length, 4))
        XYZcoord[:, :, 1:4] = bone.reshape((batchSize, length, 3))
        return XYZcoord

    def transorms(Transarray, batchSize):
        T = np.zeros((batchSize, 4, 4))
        T[:, 1:, :] = np.transpose(Transarray.reshape((batchSize, 4, 3)), (0, 2, 1))
        T = np.roll(T, 1, axis=2)
        T[:, 0, 0] = 1
        return T

    batchSize = X.shape[0]
    femX = X[:, 0:7152]
    tibX = X[:, 10752:14055]
    patX = X[:, 15567:16983]
    TransPat = X[:, 18165:18177]
    TransTib = X[:, 18153:18165]

    femTemplate = organize(femX, batchSize)
    tibTemplate = organize(tibX, batchSize)
    patTemplate = organize(patX, batchSize)

    Tt = transorms(TransTib, batchSize)
    Tp = transorms(TransPat, batchSize)

    # Create LCS for each bone
    cs_global = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]])
    cs_fem = cs_tib = cs_pat = cs_global

    # Reorient transformations
    Tall = np.identity(4)
    Tall[[1, 2], :] = -Tall[[1, 2], :]
    Tall = np.repeat(Tall[np.newaxis, :, :], batchSize, axis=0)

    Ttrans = np.identity(4)
    Ttrans = np.repeat(Ttrans[np.newaxis, :, :], batchSize, axis=0)
    Ttrans[:, 1:3, 0] = -Tt[:, 1:3, 0]

    Trot = Ttrans
    Trot[:, 3, 0] = -Trot[:, 3, 0]

    Tt = np.einsum('ijk,ikl,ilm,imn,inp->ijp', Trot, Tall, Ttrans, Tt, Tall)
    Tp = np.einsum('ijk,ikl,ilm,imn,inp->ijp', Trot, Tall, Ttrans, Tp, Tall)

    femTemplate = np.matmul(Tall[0, :, :], np.einsum('ijk->ikj', femTemplate))
    tibTemplate = np.matmul(Tall[0, :, :], np.einsum('ijk->ikj', tibTemplate))
    patTemplate = np.matmul(Tall[0, :, :], np.einsum('ijk->ikj', patTemplate))

    # Model Alignment
    femModel = femTemplate[:, 1:4, :]
    # TODO fix this
    femCS = cs_fem

    tibModel = np.einsum('ijk,ikl->ijl', Tt, tibTemplate)[:, 1:4, :]  # np.matmul(Tt, tibTemplate)[:,1:4,:]
    tibCS = np.matmul(Tt, cs_tib.T)

    patModel = np.einsum('ijk,ikl->ijl', Tp, patTemplate)[:, 1:4, :]  # np.matmul(Tp, patTemplate)[1:4].T
    patCS = np.matmul(Tp, cs_pat.T)

    XYZModel = {}
    CS = {}

    for i in ('femModel', 'tibModel', 'patModel'):
        XYZModel[i] = locals()[i]

    for i in ('femCS', 'tibCS', 'patCS'):
        CS[i] = locals()[i]

    return XYZModel, CS


def CreateTriangleMesh(XYZModel, triangulation):
    allVertices = np.concatenate((XYZModel['femModel'],
                                  XYZModel['tibModel'], XYZModel['patModel']), axis=2)
    allVertices = np.einsum('ijk->ikj', allVertices)

    femTriangulation = triangulation[0]
    tibTriangulation = triangulation[1]
    patTriangulation = triangulation[2]

    tibTriangulation = tibTriangulation + femTriangulation.max() + 1
    patTriangulation = patTriangulation + tibTriangulation.max() + 1
    allFaces = np.vstack((femTriangulation, tibTriangulation, patTriangulation))

    return allVertices, allFaces


def DisplayModel(model, allFaces):
    # from mayavi import mlab
    # trimesh = mlab.triangular_mesh(model[0], model[1], model[2], allFaces)
    # mlab.axes(trimesh, xlabel='x', ylabel='y',x_axis_visibility=True, z_axis_visibility=True)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(model[:, 0], model[:, 1], model[:, 2], c='r', marker='o')
    ax.plot_trisurf(model[0], model[1], model[2],
                    triangles=allFaces)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def DisplayPointModel(model, scene, hold=False):
    import matplotlib.pyplot as plt
    model = model[np.random.randint(model.shape[0], size=3957), :]
    if hold:
        ax = plt.gca()
        ax.scatter(model[:, 0], model[:, 1], model[:, 2], c='b', marker='.')
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(model[:, 0], model[:, 1], model[:, 2], c='r', marker='.')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # Source (Origin)
        ax.scatter([0], [0], [0], color="g", s=500)

        # Detector Screen
        f = np.int(scene['fieldSize'] / 2)
        s = np.int(scene['src_to_screen'])
        zz, yy = np.meshgrid(range(-f, f), range(-f, f))
        xx = zz * 0 + s
        ax.plot_surface(xx, yy, zz)


def Display2DPoints(model):
    import matplotlib.pyplot as plt
    model = model[np.random.randint(model.shape[0], size=3957), :]
    ax = plt.gca()
    ax.scatter(model[:, 0], model[:, 1], c='r', marker='.')


def TileImages(hist, tiles, shapeParams):
    totalPoses = hist.shape[0]
    rows = int(np.sqrt(tiles))
    cols = rows
    totalSets = int(totalPoses / tiles)
    sets = np.zeros([totalSets, tiles])
    multiplicationFactor = 10
    shapeParamsOffset = 10
    for set in range(totalSets):
        sets[set, :] = np.linspace(set, totalPoses - totalSets + set, tiles)

    tiledImage = np.zeros([totalSets, int(256 * np.sqrt(tiles)), int(256 * np.sqrt(tiles) + 1)])
    for set in range(totalSets):
        n = 0
        for row in range(rows):
            for col in range(cols):
                tiledImage[set, row * 256:(row + 1) * 256, col * 256:(col + 1) * 256] = multiplicationFactor * hist[
                    int(sets[set, n])]
                n += 1
        tiledImage[set, 0:len(shapeParams),
        tiledImage.shape[2] - 1] = (shapeParams + shapeParamsOffset) * 1000

    return tiledImage
