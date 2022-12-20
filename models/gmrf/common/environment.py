import numpy as np
from models.gmrf.common.map import DiscreteMap, DiscreteMap, DiscreteScalarMap, DiscreteScalarMap, DiscreteVectorMap
import pandas as pd
from models.gmrf.common.observation import Observation
from models.gmrf.common.lattice import Lattice, LatticeScalar
from models.gmrf.utils.report.plot import plotScalarField
import copy
import os
import scipy.sparse.linalg
import scipy.sparse
from .obstacle_map import ObstacleMap


#   Visual representation of coordinate systems (x,y) vs (i,j)
#
#       -----> j
#       |	+------------------+
#       |	|                  |
#       V	|                  |
#       i	|                  |
#           |     LATTICE      |
#           |                  |
#       y	|                  |
#       ^	|                  |
#       |	|                  |
#       |	+------------------+
#       -----> x
#

##==============================================================================
class EnvironmentGroundTruth(DiscreteMap):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions=2, size=(0.0, 0.0), resolution=1.0, offset=0):

        assert dimensions == 2, dimensions  # For now

        DiscreteMap.__init__(self, dimensions=dimensions, size=size, resolution=resolution, offset=offset)

        self.gas = DiscreteScalarMap(dimensions=dimensions, size=size, resolution=resolution, offset=offset)
        self.wind = DiscreteVectorMap(dimensions=dimensions, size=size, resolution=resolution, offset=offset)
        self.obstacles = ObstacleMap(dimensions=dimensions, size=size, resolution=resolution, offset=offset)


        return

    @classmethod
    def fromGadenCsv(cls, gaden_csv_file, height=0):

        ### GET METADATA OF ENVIRONMENT ###
        resolution = -1.0
        with open(gaden_csv_file) as f:

            # Environment size
            line = f.readline()
            min_x = line.split()[1]
            min_y = line.split()[2]
            line = f.readline()
            max_x = line.split()[1]
            max_y = line.split()[2]
            size_x = float(max_x) - float(min_x)
            size_y = float(max_y) - float(min_y)
            size = (size_x, size_y)

            # Shape
            line = f.readline()
            cells_j = int(line.split()[1])
            cells_i = int(line.split()[2])
            shape = (cells_i, cells_j)

            # Resolution
            line = f.readline()
            resolution = float(line.split()[1])

        # Init instance
        assert resolution > 0
        assert type(size) == tuple
        instance = cls(dimensions=2, size=size, resolution=resolution)
        assert (instance.shape == shape)
        assert (type(instance) == EnvironmentGroundTruth)

        ### READ DATA ###
        data = pd.read_csv(gaden_csv_file, sep=' ', header=6)
        data = data[data["Cell_z\t"] == height]

        for index, row in data.iterrows():
            j = int(row["Cell_x\t"])
            i_alt = int(row["Cell_y\t"])
            i = cells_i - i_alt - 1
            gas = row["Gas_conc[ppm]\t"]
            wind = (-row["Wind_v[m/s]\t"], row["Wind_u[m/s]\t"])

            if gas == 0.0 and wind[0] == 0 and wind[1] == 0.0:
                obstacle = 1
            else:
                obstacle = 0

            cell = (i, j)
            instance.gas.setCell(cell, gas)
            instance.wind.setCell(cell, wind)
            instance.obstacles.setCell(cell, obstacle)

        assert (instance.obstacles.max() <= 1.0)
        assert (instance.obstacles.min() >= 0.0)

        return instance

    ## METHODS -----------------------------------------------------------------

    def getObservation(self, position):
        return self.getPosition(position)

    ## BASE CLASS IMPLEMENTATION -----------------------------------------------

    def _getCell(self, cell):
        gas = self.gas.getCell(cell)
        wind_ij = self.wind.getCell(cell)
        wind_xy = (wind_ij[1], -wind_ij[0])
        obstacle = self.obstacles.getCell(cell)
        position = self._convertCellToPosition(cell)
        o = Observation(position=position, gas=gas, wind=wind_xy, time=0, dimensions=2, data_type='gas+wind')
        return o

    def _setCell(self, cell, value):
        assert (False)  # This has still to be implemented

    """
    def _getPosition(self, position):
        gas = self.gas.getPosition(position)
        wind = self.wind.getPosition(position)
        self.wind.plot()
        assert False, "Comprobar si hay que cambiar coordenadas del viento de IJ a XY, creo que si"
        obstacle = self.obstacles.getPosition(position)
        position = position
        o = Observation(position, gas, wind, 0, 2, 'gas+wind')
        return o
    """

    def plot(self):
        self.gas.plot()
        self.wind.plot()
        self.obstacles.plot()

    def _normalize(self):
        self.gas.normalize()
        self.wind.normalize()
        self.obstacles.normalize()


##==============================================================================
class InterCellScalarMap(DiscreteMap):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions, size=(0.0, 0.0), resolution=1.0):
        assert (resolution > 0.0)
        DiscreteMap.__init__(self, dimensions=dimensions, size=size, resolution=resolution)
        return

    ## Base class implementation -----------------------------------------------

    def _getCell(self, cell):
        assert False

    def _setCell(self, cell, value):
        assert False

    def plot(self):
        assert False

    def _normalize(self):
        assert False

    def _getMaxCell(self):
        assert False


##==============================================================================
class EnvironmentRunningGroundTruth(DiscreteMap):

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, dimensions=2, size=(0.0, 0.0), resolution=1.0, period=1.0, offset=0):

        assert dimensions == 2  # For now

        DiscreteMap.__init__(self, dimensions, size, resolution, offset=offset)

        self.environments = [EnvironmentGroundTruth(dimensions=dimensions, size=size, resolution=resolution, offset=offset)]
        self.period = period
        self.gas = DiscreteScalarMap(self.dimensions, self.size, self.resolution, offset=self.offset)
        self.wind = DiscreteVectorMap(self.dimensions, self.size, self.resolution, offset=self.offset)
        self.obstacles = ObstacleMap(self.dimensions, self.size, self.resolution, offset=self.offset)

        assert self.gas.dimensions == self.wind.dimensions == self.obstacles.dimensions

        return

    @classmethod
    def fromGadenCsvFolder(cls, gaden_csv_folder, height=0, period=1):

        environments = []

        ## FOR EACH GADEN FILE -> CREATE ENVIRONMENT
        for file in os.listdir(gaden_csv_folder):
            gaden_csv_file = gaden_csv_folder + "/" + file
            if os.path.isfile(gaden_csv_file):

                ### GET METADATA OF ENVIRONMENT ###
                resolution = -1.0
                with open(gaden_csv_file) as f:

                    # Environment size
                    line = f.readline()
                    min_x = line.split()[1]
                    min_y = line.split()[2]
                    line = f.readline()
                    max_x = line.split()[1]
                    max_y = line.split()[2]
                    size_x = float(max_x) - float(min_x)
                    size_y = float(max_y) - float(min_y)
                    size = (size_x, size_y)

                    # Shape
                    line = f.readline()
                    cells_j = int(line.split()[1])
                    cells_i = int(line.split()[2])
                    shape = (cells_i, cells_j)

                    # Resolution
                    line = f.readline()
                    resolution = float(line.split()[1])

                # Init instance
                assert resolution > 0
                environment = EnvironmentGroundTruth(dimensions=2, size=size, resolution=resolution)
                assert (environment.shape == shape)
                assert (type(environment) == EnvironmentGroundTruth)

                ### READ DATA ###
                data = pd.read_csv(gaden_csv_file, sep=' ', header=6)
                data = data[data["Cell_z\t"] == height]

                for index, row in data.iterrows():
                    j = int(row["Cell_x\t"])
                    i_alt = int(row["Cell_y\t"])
                    i = cells_i - i_alt - 1
                    gas = row["Gas_conc[ppm]\t"]
                    wind = (-row["Wind_v[m/s]\t"], row["Wind_u[m/s]\t"])

                    if gas == 0.0 and wind[0] == 0 and wind[1] == 0.0:
                        obstacle = 1
                    else:
                        obstacle = 0

                    assert -10 < wind[0] < 10
                    assert -10 < wind[1] < 10

                    cell = (i, j)
                    environment.gas.setCell(cell, gas)
                    environment.wind.setCell(cell, wind)
                    environment.obstacles.setCell(cell, obstacle)

                assert (environment.obstacles.max() <= 1.0)
                assert (environment.obstacles.min() >= 0.0)

                environments += [environment]

        ## COMBINE ALL ENVIRONMENTS
        dimensions = 2
        size = environments[0].size
        resolution = environments[0].resolution
        assert type(environments) is list and type(environments[0]) is EnvironmentGroundTruth
        for e in environments:
            assert e.size == size
            assert e.resolution == resolution

        instance = cls(dimensions, size, resolution, period)
        instance.environments = environments

        gass = [e.gas.toMatrix() for e in environments]
        winds = [e.wind.toMatrix() for e in environments]
        windsi = [wind[0] for wind in winds]
        windsj = [wind[1] for wind in winds]
        obstacless = [e.obstacles.toMatrix() for e in environments]
        assert len(gass) == len(windsi) == len(obstacless)
        n = len(gass)
        gas = gass[0]
        windi = windsi[0]
        windj = windsj[0]
        obstacles = obstacless[0]
        for i in range(1, n):
            gas += gass[i]
            windi += windsi[i]
            windj += windsj[i]
            obstacles += obstacless[i]
        instance.gas.loadMatrix(gas / n)
        instance.wind.loadMatrix((windi / n, windj / n))
        instance.obstacles.loadMatrix(obstacles / n)

        return instance

    ## METHODS -----------------------------------------------------------------

    def getObservation(self, position, time=0):
        len_data = len(self.environments)
        index = int((time * self.period) % len_data)
        return self.environments[index].getPosition(position)

    ## BASE CLASS IMPLEMENTATION -----------------------------------------------

    def _getCell(self, cell, time=0):
        len_data = len(self.environments)
        index = int((time * self.period) % len_data)
        return self.environments[index].getCell(cell)

    def _setCell(self, cell, value):
        assert (False)  # This has still to be implemented

    """
    def _getPosition(self, position):
        gas = self.gas.getPosition(position)
        wind = self.wind.getPosition(position)
        self.wind.plot()
        assert False, "Comprobar si hay que cambiar coordenadas del viento de IJ a XY, creo que si"
        obstacle = self.obstacles.getPosition(position)
        position = position
        o = Observation(position, gas, wind, 0, 2, 'gas+wind')
        return o
    """

    def plot(self, time=-1):
        if time < 0:
            self.gas.plot()
            self.wind.plot()
            self.obstacles.plot()
        else:
            len_data = len(self.environments)
            index = int((time * self.period) % len_data)
            self.environments[index].plot()
        return self

    def _normalize(self):
        self.gas.normalize()
        self.wind.normalize()
        self.obstacles.normalize()
        for e in self.environments:
            e.gas.normalize()
            e.wind.normalize()
            e.obstacles.normalize()
        return self
