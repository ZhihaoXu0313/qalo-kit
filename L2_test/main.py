from qalo.utils import extract_composition, binvec2poscar
from qalo.module.mlp import *

from lammps import lammps 
import numpy as np 
from pymatgen.io.vasp import Poscar 
from pymatgen.io.lammps.data import LammpsData 
import os


def mlp_test(poscar, outcar):
    poscar = "/afs/crc.nd.edu/user/z/zxu8/projects/qalo-kit/examples/database/dft/data-2x2x2/s-0/POSCAR"
    outcar = "/afs/crc.nd.edu/user/z/zxu8/projects/qalo-kit/examples/database/dft/data-2x2x2/s-0/OUTCAR"
    data = "/afs/crc.nd.edu/user/z/zxu8/projects/qalo-kit/examples/tmp/NbMoTaW.data"
    infile = "/afs/crc.nd.edu/user/z/zxu8/projects/qalo-kit/examples/input/in.snap.lmp"
    coeffile = "/afs/crc.nd.edu/user/z/zxu8/projects/qalo-kit/examples/input/NbMoTaW.snapcoeff"
    composition = extract_composition(poscar)
    poscar2data(poscar, data)
    eref = eform2tot(composition, calculate_pe(infile, coeffile))
    print(eref)