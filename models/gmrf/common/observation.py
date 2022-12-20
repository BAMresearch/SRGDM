import numpy as np


class Observation:
    def __init__(self,
                 position=(0.0, 0.0),
                 gas=0.0,
                 wind=(0.0, 0.0),
                 time=0,
                 data_type='gas',
                 variance_gas=0,
                 variance_wind=0,
                 dimensions=2):

        assert type(dimensions) == int and dimensions >= 1
        assert len(position) == dimensions, str(len(position)) + " " + str(dimensions)
        assert len(wind) == dimensions, str(len(wind)) + " " + str(dimensions)
        assert gas >= -0.00001
        assert type(wind) is tuple, str(wind)
        assert (data_type == 'gas' or data_type == 'wind' or data_type == 'gas+wind')
        assert variance_gas >= 0
        assert variance_wind >= 0

        self.position = position
        self.gas = gas
        self.wind = wind
        self.time = time
        self.variance_gas = variance_gas
        self.variance_wind = variance_wind

        if data_type == 'gas':
            self.data_type = 'g'
        elif data_type == 'wind':
            self.data_type = 'w'
        elif data_type == 'gas+wind':
            self.data_type = '2'
        else:
            self.data_type = ''

        return

    def __str__(self):
        return str("[P: (" + str(np.round(self.position[0], 3))
                   + ", " + str(np.round(self.position[1], 3)) + ")"
                   + "\tGas: " + str(np.round(self.gas, 3))
                   + "\tWx: " + str(np.round(self.wind[0], 3))
                   + "\tWy: " + str(np.round(self.wind[1], 3))
                   + "\ttype: " + str(self.data_type)
                   + "]")

    def hasGas(self):
        return self.data_type == 'g' or self.data_type == '2'

    def hasWind(self):
        return self.data_type == 'w' or self.data_type == '2'
