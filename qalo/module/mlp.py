from qalo.utils import extract_composition, binvec2poscar

from lammps import lammps 
import numpy as np 
from pymatgen.io.vasp import Poscar 
from pymatgen.io.lammps.data import LammpsData 
import os


def SNAPLammpsObj(lmp, infile):
    lmp.file(infile)

    natoms = lmp.get_natoms()
    types = lmp.gather_atoms("type", 0, 1)

    bs_array = lmp.extract_compute("SNA", 1, 2)
    return natoms, types, bs_array


def read_snap_coeff(coeffile):
    fh = open(coeffile, 'r')
    line = fh.readline()
    line = fh.readline()
    line = fh.readline()
    line = fh.readline()
    ntypes = [int(x) for x in line.split()][0]
    ncoeff = [int(x) for x in line.split()][1]
    coeffs = np.zeros([ntypes, ncoeff])
    for t in range(0,ntypes):
        line = fh.readline()
        for c in range(0, ncoeff):
            line = fh.readline()
            coeffs[t][c] = float(line)
    # print(coeffs)
    fh.close()
    return coeffs, ncoeff


def poscar2data(poscar, data):
    structure = Poscar.from_file(poscar).structure
    lammps_data = LammpsData.from_structure(structure, atom_style="atomic")
    lammps_data.write_file(data)


def eform2tot(composition, snape):
    dE = -composition[0] * 10.2002 - composition[1] * 10.9396 - composition[2] * 11.7814 - composition[3] * 12.9913
    return dE + snape


def calculate_pe(infile, coeffile):
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    natoms, types, bs_array = SNAPLammpsObj(lmp, infile)
    coeffs, ncoeff = read_snap_coeff(coeffile)

    pe = 0.
    for n in range(0, natoms):
        t = types[n] - 1
        ei = coeffs[types[n] - 1][0]
        for k in range(1, ncoeff):
            ei += coeffs[t][k] * bs_array[n][k - 1]
        pe += ei
    lmp.close()
    return pe


def snap_model_inference(binvec, infile, elements, alat, nsites, spc_size, coeffile, path_of_tmp):
    pPOSCAR = os.path.join(path_of_tmp, "POSCAR")
    pData = os.path.join(path_of_tmp, "NbMoTaW.data")
    if os.path.exists(pPOSCAR):
        os.remove(pPOSCAR)
    if os.path.exists(pData):
        os.remove(pData)

    binvec2poscar(binvec=binvec, poscar=pPOSCAR, elements=elements, alat=alat, nsites=nsites, spc_size=spc_size)
    composition = extract_composition(pPOSCAR)
    poscar2data(pPOSCAR, pData)
    e = eform2tot(composition, calculate_pe(infile, coeffile))

    os.remove(pPOSCAR)
    os.remove(pData)
    return e

