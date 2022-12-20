from .observation import Observation
from .map import Map, DiscreteMap, DiscreteScalarMap, DiscreteVectorMap
from .probability_dist import MultivariateNormalPDFMap
from abc import abstractmethod
from .lattice import LatticeScalar


################################################################################
## GENERIC DISTRIBUTION MAPPER
################################################################################

##==============================================================================
class DistributionMapper:

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions, size):
        self._observations = []
        self.__has_new_observations = False
        self.__is_estimate_valid = False
        self.dimensions = dimensions
        self.size = size
        return

    ## Abstract ------------------------------------------------------------

    @abstractmethod
    def _updateObservations(self):
        pass

    @abstractmethod
    def _estimate(self):
        pass

    def _isValidMap(self, map):
        return isinstance(map, Map) and map.dimensions == self.dimensions and map.size == self.size

    ## METHODS -----------------------------------------------------------------

    def addObservation(self, observation):
        if type(observation) is list and len(observation) > 0:
            assert (type(observation[0]) is Observation)
            self._observations += observation

        elif type(observation) is Observation:
            self._observations += [observation]

        else:
            print("[GDM.addObservation()] Got an invalid observation. I will simply ignore it")
            return self

        self.__has_new_observations = True
        self._updateObservations()
        return self

    def estimate(self):
        """ Update prediction if new observations become available """
        if self.__has_new_observations or not self.__is_estimate_valid:
            if len(self._observations) > 0:
                self._estimate()
                self.__has_new_observations = False
                self.__is_estimate_valid = True
        return self


##==============================================================================
class DiscreteDistributionMapper(DistributionMapper):
    def __init__(self, dimensions, size, resolution):
        DistributionMapper.__init__(self, dimensions=dimensions, size=size)
        self.resolution = resolution
        return

    def _isValidMap(self, map):
        return DistributionMapper._isValidMap(self, map) and isinstance(map, DiscreteMap)



##==============================================================================
class ProbabilisticDistributionMapper(DistributionMapper):

    ## CONSTRUCTORS ------------------------------------------------------------
    def __init__(self, dimensions, size):
        DistributionMapper.__init__(self, dimensions=dimensions, size=size)
        self.__is_uncertainty_valid = False
        return

    ## Abstract ------------------------------------------------------------

    @abstractmethod
    def _computeUncertainty(self):
        pass

    ## METHODS -----------------------------------------------------------------

    def computeUncertainty(self):
        self.estimate()
        if not self.__is_uncertainty_valid and len(self._observations) > 0:
            self._computeUncertainty()
            self.__is_uncertainty_valid = True
        return self

    # uncertainty = self._getUncertainty()
    # assert isinstance(uncertainty, Map) or (type(uncertainty) is tuple and isinstance(uncertainty[0], Map))
    # return uncertainty

    ## BASE CLASS IMPLEMENTATION -----------------------------------------------
    def addObservation(self, observation):
        self.__is_uncertainty_valid = False
        return DistributionMapper.addObservation(self, observation)


##==============================================================================
class NormalDistributionMapper(ProbabilisticDistributionMapper,
                               DiscreteDistributionMapper):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions, size, resolution,):
        ProbabilisticDistributionMapper.__init__(self, dimensions=dimensions, size=size)
        DiscreteDistributionMapper.__init__(self, dimensions=dimensions, size=size, resolution=resolution)
        return

    ## Abstract ------------------------------------------------------------

    @abstractmethod
    def _toNormalDistributionMap(self):
        pass

    ## METHODS -----------------------------------------------------------------

    def toNormalDistributionMap(self):
        pdf = self._toNormalDistributionMap()
        assert self._isValidPDFMap(pdf)
        return pdf

    def _isValidPDFMap(self, pdf):
        return isinstance(pdf, MultivariateNormalPDFMap) or (
                    type(pdf) is tuple and isinstance(pdf[0], MultivariateNormalPDFMap))


################################################################################
## GAS
################################################################################


##==============================================================================
class GasDistributionMapper(DistributionMapper):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions, size):
        DistributionMapper.__init__(self, dimensions=dimensions, size=size)
        assert type(self) is not GasDistributionMapper
        return

    ## Abstract ------------------------------------------------------------

    @abstractmethod
    def _getGasEstimate(self):
        pass

    ## METHODS -----------------------------------------------------------------

    def getGasEstimate(self):
        self.estimate()
        estimate = self._getGasEstimate()
        assert self._isValidMap(estimate)
        return estimate


##==============================================================================
class DiscreteGasDistributionMapper(DiscreteDistributionMapper,
                                    GasDistributionMapper):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions, size, resolution):
        ## Enforce derived attributes
        assert type(self) is not DiscreteGasDistributionMapper

        ## Init base
        DiscreteDistributionMapper.__init__(self, dimensions=dimensions, size=size, resolution=resolution)
        GasDistributionMapper.__init__(self, dimensions=dimensions, size=size)

        ## Member variables
        #self._gas = DiscreteScalarMap(dimensions=2, size=size, resolution=resolution, init_value=0)

        return

    ## Abstract ------------------------------------------------------------

    @abstractmethod
    def _getGasEstimate(self):
        pass


##==============================================================================
class ProbabilisticGasDistributionMapper(ProbabilisticDistributionMapper,
                                         GasDistributionMapper):

    ## CONSTRUCTORS ------------------------------------------------------------
    def __init__(self, dimensions, size):
        ProbabilisticDistributionMapper.__init__(self, dimensions=dimensions, size=size)
        GasDistributionMapper.__init__(self, dimensions=dimensions, size=size)
        assert type(self) is not ProbabilisticGasDistributionMapper
        return

    ## Abstract ------------------------------------------------------------

    @abstractmethod
    def _getGasUncertainty(self):
        pass

    ## METHODS -----------------------------------------------------------------

    def getGasUncertainty(self):
        self.computeUncertainty()
        uncertainty = self._getGasUncertainty()
        assert self._isValidMap(uncertainty)
        return uncertainty


##==============================================================================
class NormalGasDistributionMapper(DiscreteGasDistributionMapper,
                                  NormalDistributionMapper,
                                  ProbabilisticGasDistributionMapper):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions, size, resolution):
        ## Base init
        DiscreteGasDistributionMapper.__init__(self, dimensions=dimensions, size=size, resolution=resolution)
        NormalDistributionMapper.__init__(self, dimensions=dimensions, size=size, resolution=resolution)
        ProbabilisticGasDistributionMapper.__init__(self, dimensions=dimensions, size=size)

        ## Member variables
        #self._gas_uncertainty = DiscreteScalarMap(dimensions=2, size=size, resolution=resolution,
        #                                         init_value=float("+inf"))

        return

    ## BASE CLASS IMPLEMENTATION -----------------------------------------------

    def _toNormalDistributionMap(self):
        gas_mean = self.getGasEstimate()
        gas_var = self.getGasUncertainty()
        pdf = MultivariateNormalPDFMap(gas_mean, gas_var, self.resolution)
        return pdf


################################################################################
## WIND
################################################################################

##==============================================================================
class WindDistributionMapper(DistributionMapper):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions, size):
        DistributionMapper.__init__(self, dimensions=dimensions, size=size)
        assert type(self) is not WindDistributionMapper
        return

    ## Abstract ------------------------------------------------------------

    @abstractmethod
    def _getWindEstimate(self):
        pass

    ## METHODS -----------------------------------------------------------------

    def getWindEstimate(self):
        self.estimate()
        estimate = self._getWindEstimate()
        assert self._isValidMap(estimate)
        return estimate


##==============================================================================
class DiscreteWindDistributionMapper(DiscreteDistributionMapper,
                                     WindDistributionMapper):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions, size, resolution):
        ## Init base
        DiscreteDistributionMapper.__init__(self, dimensions=dimensions, size=size, resolution=resolution)
        WindDistributionMapper.__init__(self, dimensions=dimensions, size=size)
        assert type(self) is not DiscreteWindDistributionMapper

        ## Member variables
        #self._wind = DiscreteVectorMap(dimensions=2, size=size, resolution=resolution)
        return

    ## Abstract ------------------------------------------------------------

    @abstractmethod
    def _getWindEstimate(self):
        pass


##==============================================================================
class ProbabilisticWindDistributionMapper(ProbabilisticDistributionMapper,
                                          WindDistributionMapper):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions, size):
        ProbabilisticDistributionMapper.__init__(self, dimensions=dimensions, size=size)
        WindDistributionMapper.__init__(self, dimensions=dimensions, size=size)
        return

    ## Abstract ------------------------------------------------------------

    @abstractmethod
    def _getWindUncertainty(self):
        pass

    ## METHODS -----------------------------------------------------------------

    def getWindUncertainty(self):
        self.computeUncertainty()
        uncertainty = self._getWindUncertainty()
        assert self._isValidMap(uncertainty)
        return uncertainty


##==============================================================================
class NormalWindDistributionMapper(DiscreteWindDistributionMapper,
                                   NormalDistributionMapper,
                                   ProbabilisticWindDistributionMapper):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions, size, resolution):
        ## Base init
        DiscreteWindDistributionMapper.__init__(self, dimensions=dimensions, size=size, resolution=resolution)
        NormalDistributionMapper.__init__(self, dimensions=dimensions, size=size, resolution=resolution)
        ProbabilisticWindDistributionMapper.__init__(self, dimensions=dimensions, size=size)

        ## Member variables
        #self._wind_uncertainty = DiscreteVectorMap(dimensions=2, size=size, resolution=resolution)
        #self._wind_uncertainty.fill((float("+inf"), float("+inf")))

        return

    ## BASE CLASS IMPLEMENTATION -----------------------------------------------

    def _toNormalDistributionMap(self):
        wind_mean_0, wind_mean_1 = self.getWindEstimate().toMatrix()
        wind_var_0, wind_var_1 = self.getWindUncertainty().toMatrix()
        pdf_0 = MultivariateNormalPDFMap(LatticeScalar.fromMatrix(wind_mean_0), LatticeScalar.fromMatrix(wind_var_0),
                                         self.resolution)
        pdf_1 = MultivariateNormalPDFMap(LatticeScalar.fromMatrix(wind_mean_1), LatticeScalar.fromMatrix(wind_var_1),
                                         self.resolution)

        return pdf_0, pdf_1
