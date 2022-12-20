from .lattice import LatticeScalar
from .map import DiscreteMap
from abc import abstractmethod


##==============================================================================
class PDF:

    def __init__(self):
        return

    ## Abstract ------------------------------------------------------------

    @abstractmethod
    def _getMean(self):
        pass

    @abstractmethod
    def _getVariance(self):
        pass

    ## METHODS -----------------------------------------------------------------

    def getMean(self):
        return self._getMean()

    def getVariance(self):
        return self._getVariance()


##==============================================================================
class NormalPDF(PDF):

    def __init__(self, mean, variance):
        ## Init base
        PDF.__init__(self)

        ## Member variables
        self._mean = mean
        self._variance = variance

        return

    ## BASE CLASS IMPLEMENTATION -----------------------------------------------

    def _getMean(self):
        return self._mean

    def _getVariance(self):
        return self._variance


##==============================================================================
class MultivariateNormalPDF(NormalPDF):

    def __init__(self, mean, variance):
        ## Parameter check
        assert mean.shape == variance.shape

        ## Enforce derived attributes
        assert type(self) is not MultivariateNormalPDF

        ## Init base
        NormalPDF.__init__(self, mean, variance)

        return


##==============================================================================
class MultivariateNormalPDFMap(MultivariateNormalPDF, DiscreteMap):

    def __init__(self, mean, variance, resolution, offset=0):

        ## Param check
        assert isinstance(mean, LatticeScalar)
        assert isinstance(variance, LatticeScalar)
        dimensions = mean.dimensions
        assert mean.dimensions == variance.dimensions
        for x in range(dimensions):
            assert mean.shape[x] == variance.shape[x]

        ## Aux
        size_x = mean.shape[1] * resolution
        size_y = mean.shape[0] * resolution
        size = (size_x, size_y)

        ## Init base
        NormalPDF.__init__(self, mean=mean, variance=variance)
        DiscreteMap.__init__(self, dimensions=dimensions, size=size, resolution=resolution, offset=offset)

        return

    ## BASE CLASS IMPLEMENTATION -----------------------------------------------

    def _getCell(self, cell):
        cell_mean = self._mean.getCell(cell)
        cell_variance = self._variance.getCell(cell)
        normal = NormalPDF(cell_mean, cell_variance)
        return normal

    def _setCell(self, cell, value):
        assert type(value) is NormalPDF
        cell_mean = value.getMean()
        cell_variance = value.getVariance()
        self._mean.setCell(cell, cell_mean)
        self._variance.setCell(cell, cell_variance)
        return self

    def plot(self, mean_max=0, variance_max=0, mask=0, save=""):
        if save == "":
            self._mean.plot(vmax=mean_max, mask=mask)
            self._variance.plot(vmax=variance_max, mask=mask)
        else:
            self._mean.plot(vmax=mean_max, mask=mask, save=save + "_mean")
            self._variance.plot(vmax=variance_max, mask=mask, save=save + "_var")

    def _normalize(self):
        assert False

    def _getMaxCell(self):
        assert False


if __name__ == "__main__":
    m = LatticeScalar((10, 10))
    M = MultivariateNormalPDFMap(m, m)
    M.plot()
