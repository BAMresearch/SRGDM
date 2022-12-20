import sys
sys.path.append("/home/nicolas/git/gdm/")

from common import environment
import os


script_path = os.path.dirname(os.path.realpath(__file__))
relative_csv_file = "../data/test_environments/small_office/3/data.csv"
csv_file = script_path + "/" + relative_csv_file


print("Loading small_office_3... ", end =" ")
assert os.path.isfile(csv_file)
small_office_3 = environment.EnvironmentGroundTruth.fromGadenCsv(csv_file)
print("Done")

if __name__ == "__main__":

    #small_office_3.plot()


    ## Print gradients
    from utils.report.plot import plotScalarField, plotVectorField
    import numpy as np
    #plotVectorField(, small_office_3.wind.toMatrix(1), interpol=4, magnitude=0.025*small_office_3.gas.toMatrix(), vmax=0.6, scale=15)
    gas = small_office_3.gas.toMatrix()[0:47, 0:70]
    gas_gradient = np.gradient(gas)
    wind_i = small_office_3.wind.toMatrix(0)[0:47, 0:70]
    wind_j = small_office_3.wind.toMatrix(1)[0:47, 0:70]
    plotVectorField(wind_i, wind_j, interpol=3, magnitude=gas, vmax=25, scale=12)
    plotVectorField(gas_gradient[0], gas_gradient[1], interpol=3, scale=75, magnitude=gas, vmax=25)
    obstacle = small_office_3.obstacles.toMatrix()[0:47, 0:70]
    plotScalarField(obstacle, mask=1)
