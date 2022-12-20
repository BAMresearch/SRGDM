if(__name__ == "__main__"):
    import sys
    sys.path.insert(0, "../../..")
    from gdm.common.lattice import Lattice, LatticeScalar, LatticeVector
    from typing import Dict


    l = LatticeScalar(2, (15,15))
    l.fill(-1)
    l.setCell((1, 1), 3)
    #l.normalize()
    # l.plot()
    print(l.max())
    print(l.toMatrix())
    print(l._getAllCellCoordinatesBetween( (0,0),(2,14) ))

    v = LatticeVector(2, (3,3))
    v.fill((-1,-1))
    v.setCell((1,1),(3,4))
    #v.normalize()
    #v.plot(interpol=1)

    m = v.toMatrix()
    print(len(m))
    print(m[0])
    print(m[1])
    v.loadMatrix(m)
    v.plot(interpol=1)

    print(l.getCellPattern_1x1())


