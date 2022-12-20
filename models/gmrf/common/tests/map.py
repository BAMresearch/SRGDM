if (__name__ == "__main__"):
    import sys

    sys.path.insert(0, "../../..")
    from gdm.common import DiscreteScalarMap
    import numpy as np

    """
    dsm = map.DiscreteScalarMap((10,5),0.1)
    print(dsm.size)
    print(dsm.getCell((1,1)))
    
    
    print("----------------")
    m = np.array(((1,2,3),(4,5,6)))
    dsm = map.DiscreteScalarMap.fromMatrix(m, 0.1)
    print(dsm.size)
    print(dsm.shape)
    print(dsm.toMatrix())
    dsm.plot()
    
    
    print("----------------")
    dsm = map.DiscreteScalarMap.fromPGM("../data/test_environments/small_office/3/obstacles.pgm", 0.1)
    dsm.normalize().invertValues().plot()
    print(type(dsm))
    
    
    print("----------------")
    dvm = map.DiscreteVectorMap((10,5), 1, (1,1))
    dvm.plot()
    """

    print("----------------")
    m = np.array((((0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14), (15, 16, 17, 18, 19)))).T
    dsm = DiscreteScalarMap.fromMatrix(m, 0.1)
    print(dsm.size)
    print(dsm.shape)
    print(dsm.toMatrix())


    # dsm.plot()


    def test_convert_cell(dsm, cell):
        position = dsm._convertCellToPosition(cell)
        new_cell = dsm._convertPositionToCell(position)
        print(str(cell) + "->" + str(position) + "->" + str(new_cell))
        assert cell == new_cell


    def test_convert_position(dsm, position):
        cell = dsm._convertPositionToCell(position)
        new_position = dsm._convertCellToPosition(cell)
        print(str(position) + "->" + str(cell) + "->" + str(new_position))

        x = position[0]
        y = position[1]
        nx = new_position[0]
        ny = new_position[1]
        assert x - dsm.resolution / 2 <= nx <= x + dsm.resolution / 2
        assert y - dsm.resolution / 2 <= ny <= y + dsm.resolution / 2


    print("Cells -> Position")
    test_convert_cell(dsm, cell=(0, 0))
    test_convert_cell(dsm, cell=(4, 0))
    test_convert_cell(dsm, cell=(4, 3))
    test_convert_cell(dsm, cell=(0, 3))
    test_convert_cell(dsm, cell=(1, 1))
    test_convert_cell(dsm, cell=(3, 3))
    test_convert_cell(dsm, cell=(2, 1))
    test_convert_cell(dsm, cell=(4, 1))

    print("")
    print("Position -> Cells")
    test_convert_position(dsm, position=(0.01, 0.01))
    test_convert_position(dsm, position=(0.01, 0.49))
    test_convert_position(dsm, position=(0.39, 0.49))
    test_convert_position(dsm, position=(0.39, 0.01))
    test_convert_position(dsm, position=(0.00, 0.00))
    test_convert_position(dsm, position=(0.1, 0.1))
    test_convert_position(dsm, position=(0.25, 0.35))
    test_convert_position(dsm, position=(0.35, 0.25))

    print("----------------")
    print("Offset")
    m = np.array((((0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14), (15, 16, 17, 18, 19)))).T
    dsm = DiscreteScalarMap.fromMatrix(m, 0.1, offset=(0.12, 0.13))
    print(dsm.size)
    print(dsm.shape)
    print(dsm.toMatrix())

    print("Cells -> Position")
    test_convert_cell(dsm, cell=(0, 0))
    test_convert_cell(dsm, cell=(4, 0))
    test_convert_cell(dsm, cell=(4, 3))
    test_convert_cell(dsm, cell=(0, 3))
    test_convert_cell(dsm, cell=(1, 1))
    test_convert_cell(dsm, cell=(3, 3))
    test_convert_cell(dsm, cell=(2, 1))
    test_convert_cell(dsm, cell=(4, 1))

    print("")
    print("Position -> Cells")
    test_convert_position(dsm, position=(0.01, 0.01))
    test_convert_position(dsm, position=(-0.12, -0.13))
    test_convert_position(dsm, position=(-0.12,  0.37))
    test_convert_position(dsm, position=( 0.23,  0.37))
    test_convert_position(dsm, position=( 0.23, -0.13))

    print("----------------")
    dsm = DiscreteScalarMap(dimensions=2, size=(10.4, 10.4), resolution=0.1, offset=(0.1,0.1))
    test_convert_cell(dsm, cell=(0, 0))
    test_convert_cell(dsm, cell=(1, 1))



