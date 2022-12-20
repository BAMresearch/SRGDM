import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


##============================================================================
def plotScalarField(matrix,
                    vmax=0,
                    mask=0,
                    save="",
                    scale="lin",
                    fig_size=(8, 8)):
    assert type(matrix) is np.ndarray, "[plotScalarField] Expected numpy matrix"

    fig = plt.figure(num=None, figsize=fig_size, dpi=100, facecolor='w', edgecolor='w')
    ax = fig.add_subplot(111)
    if vmax == 0:
        vmax = matrix.max()

    vmin = matrix.min()
    # if vmin > 0.0:
    #	vmin = 0.0

    if vmin > vmax:
        vmin = vmax - 0.000001

    if scale == "lin":
        cax = ax.matshow(matrix, interpolation='nearest', vmin=vmin, vmax=vmax)  # , norm=LogNorm(0.0001,self._m.max()))
    elif scale == "log":
        cax = ax.matshow(matrix, interpolation='nearest', norm=LogNorm(0.0001, vmax))
    else:
        assert False

    if (mask == 1):
        """
        ax.spines['left'].set_position('center')
        ax.spines['right'].set_color('outward')
        ax.spines['bottom'].set_position('outward')
        ax.spines['top'].set_position('outward')
        ax.spines['right'].set_position('outward')
        ax.spines['top'].set_color('outward')
        """
        plt.axis('off')
    else:
        fig.colorbar(cax)

    if (save == ""):
        plt.show()
    else:
        fig.savefig(save)
        plt.close(fig)
    return


##============================================================================
def plotVectorField(m_i, m_j,
                    interpol=5,
                    scale=20,
                    vmax=0,
                    vmin=0,
                    mask=0,
                    save="",
                    fig_size=(8, 8),
                    magnitude=False):
    assert type(m_i) is np.ndarray, "[plotVectorField] Expected numpy matrix"
    assert type(m_j) is np.ndarray, "[plotVectorField] Expected numpy matrix"
    assert m_i.shape[0] == m_j.shape[0]
    assert m_i.shape[1] == m_j.shape[1]
    assert interpol > 0

    if type(magnitude) is bool and not magnitude:
        magnitude = np.sqrt(m_i ** 2 + m_j ** 2)
    else:
        assert type(magnitude) is np.ndarray, "[plotVectorField] Expected numpy matrix"
        assert magnitude.shape[0] == m_i.shape[0] == m_j.shape[0]
        assert magnitude.shape[1] == m_i.shape[1] == m_j.shape[1]

    cells_i, cells_j = m_i.shape
    vector_i = np.zeros((cells_i, cells_j))
    vector_j = np.zeros((cells_i, cells_j))

    if interpol == 5:
        vector_i[::interpol, ::interpol] = (m_i[0::interpol, 0::interpol] + m_i[1::interpol, 1::interpol] + m_i[
                                                                                                            2::interpol,
                                                                                                            2::interpol]) / 3
        vector_j[::interpol, ::interpol] = (m_j[0::interpol, 0::interpol] + m_j[1::interpol, 1::interpol] + m_j[
                                                                                                            2::interpol,
                                                                                                            2::interpol]) / 3
    else:
        vector_i[::interpol, ::interpol] = m_i[::interpol, ::interpol]
        vector_j[::interpol, ::interpol] = m_j[::interpol, ::interpol]

    if vmax == 0:
        vmax = magnitude.max()

    fig = plt.figure(num=None, figsize=fig_size, dpi=100, facecolor='w', edgecolor='k')
    plt.imshow(magnitude, extent=[0, cells_j - 1, cells_i - 1, 0], cmap='Blues', interpolation='nearest', vmax=vmax, vmin=vmin)
    plt.quiver(vector_j, -vector_i, scale=scale, minlength=0.001)

    if (mask == 1):
        """
        plt.axes().spines['left'].set_position('none')
        plt.axes().spines['right'].set_color('none')
        plt.axes().spines['bottom'].set_position('none')
        plt.axes().spines['top'].set_position('none')
        plt.axes().spines['right'].set_position('none')
        plt.axes().spines['top'].set_color('none')
        """
        plt.axis('off')

    if (save == ""):
        plt.show()
    else:
        fig.savefig(save)
        plt.close(fig)
    return


##============================================================================
def plotScalarVectorFields(m_a, m_i, m_j,
                           interpol=3,
                           scale=20,
                           vmax=0.0,
                           save="",
                           fig_size=(8, 8)):
    assert type(m_a) is np.ndarray, "[plotScalarVectorFields] Expected numpy matrix"
    assert type(m_i) is np.ndarray, "[plotScalarVectorFields] Expected numpy matrix"
    assert type(m_j) is np.ndarray, "[plotScalarVectorFields] Expected numpy matrix"

    magnitude = m_a
    cells_i, cells_j = m_i.shape
    vector_i = np.zeros((cells_i, cells_j))
    vector_j = np.zeros((cells_i, cells_j))

    if interpol == 3:
        vector_i[::interpol, ::interpol] = (m_i[0::interpol, 0::interpol] + m_i[1::interpol, 1::interpol] + m_i[
                                                                                                            2::interpol,
                                                                                                            2::interpol]) / 3
        vector_j[::interpol, ::interpol] = (m_j[0::interpol, 0::interpol] + m_j[1::interpol, 1::interpol] + m_j[
                                                                                                            2::interpol,
                                                                                                            2::interpol]) / 3
    else:
        vector_i[::interpol, ::interpol] = m_i[::interpol, ::interpol]
        vector_j[::interpol, ::interpol] = m_j[::interpol, ::interpol]

    if vmax == 0.0:
        vmax = magnitude.max()

    fig = plt.figure(num=None, figsize=fig_size, dpi=100, facecolor='w', edgecolor='k')
    plt.imshow(magnitude, extent=[0, cells_j - 1, cells_i - 1, 0], cmap='Blues', interpolation='nearest', vmax=vmax)
    plt.quiver(vector_j, -vector_i, scale=scale, minlength=0.001)

    if (save == ""):
        plt.show()
    else:
        fig.savefig(save)
        plt.close(fig)
    return


##============================================================================
def plotScalar3DField(matrix,
                      vmax=0,
                      mask=0,
                      save="",
                      scale="lin",
                      fig_size=(8, 8)):
    assert type(matrix) is np.ndarray, "[plotScalarField] Expected numpy matrix"
    x, y, z = matrix.nonzero()
    c = matrix[matrix.nonzero()]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.scatter(x, y, z, c=c, s=c ** 2, zdir='z')
    plt.show()

    if (save == ""):
        plt.show()
    else:
        fig.savefig(save)
        plt.close(fig)
    return
