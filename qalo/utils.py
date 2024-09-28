import numpy as np
import math
from pymatgen.core import Structure
import os
import csv
import re
import random


def idx2coord(idx, site_positions, supercell):
    Nx, Ny, Nz = supercell[0], supercell[1], supercell[2]
    cellIndex, siteIndex = divmod(idx, len(site_positions))

    x_unit = cellIndex % Nx
    y_unit = (cellIndex // Nx) % Ny
    z_unit = cellIndex // (Nx * Ny)

    x_site, y_site, z_site = site_positions[siteIndex]
    x = (x_unit + x_site) / Nx
    y = (y_unit + y_site) / Ny
    z = (z_unit + z_site) / Nz

    return x, y, z


def coord2idx(x, y, z, site_positions, supercell):
    Nx, Ny, Nz = supercell[0], supercell[1], supercell[2]
    x_unit = math.floor(x * Nx)
    y_unit = math.floor(y * Ny)
    z_unit = math.floor(z * Nz)

    unit_cell_index = x_unit + (y_unit * Nx) + (z_unit * Nx * Ny)
    try:
        site_index = site_positions.index([round(x * Nx, 2) % 1, round(y * Ny, 2) % 1, round(z * Nz, 2) % 1])
    except ValueError:
        raise Exception("Invalid site position")
    return int(unit_cell_index * len(site_positions) + site_index)


def distance(idx1, idx2, site_positions, supercell):
    x1, y1, z1 = idx2coord(idx1, site_positions, supercell)
    x2, y2, z2 = idx2coord(idx2, site_positions, supercell)
    dx = abs(x1 - x2) if abs(x1 - x2) <= 0.5 else 1 - abs(x1 - x2)
    dy = abs(y1 - y2) if abs(y1 - y2) <= 0.5 else 1 - abs(y1 - y2)
    dz = abs(z1 - z2) if abs(z1 - z2) <= 0.5 else 1 - abs(z1 - z2)
    r = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return r


def extract_composition(poscar):
    structure = Structure.from_file(poscar)
    composition = structure.composition
    composition_list = []
    for element, count in composition.get_el_amt_dict().items():
        composition_list.append(count)
    return composition_list


def extract_toten(outcar):
    if not os.path.exists(outcar):
        raise FileNotFoundError(f"File not found: {outcar}")
    with open(outcar, 'r') as file:
        content = file.read()

    matches = re.findall(r'TOTEN\s*=\s*(-?\d+\.\d+)', content)
    if matches:
        return matches[-1]
    else:
        return None


def save_energy(csvpath, energy, composition):
    if not isinstance(energy, (list, tuple)):
        raise ValueError("Energy must be a list or tuple")
    with open(csvpath, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(energy + composition)


def stack_dataframe(df, df_stack):
    if df_stack.empty:
        df_stack = df
    else:
        try:
            df_stack = pd.concat([df_stack, df], ignore_index=True)
        except Exception as e:
            print(e)
    return df_stack


# convert format
def poscar2binvec(poscar, spcOriginal, spcDesign, unit_site, flatten=False):
    structure = Structure.from_file(poscar)
    scaling_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    if spcOriginal != spcDesign:
        scaling_matrix = [[int(spcDesign[0] // spcOriginal[0]), 0, 0],
                          [0, int(spcDesign[1] // spcOriginal[1]), 0],
                          [0, 0, int(spcDesign[2] // spcOriginal[2])]]
        structure.make_supercell(scaling_matrix)
    element_list = list(dict.fromkeys([s.species_string for s in structure]))
    binvec = np.zeros((len(element_list), len(structure)), dtype=int)
    for index, site in enumerate(structure):
        element = site.species_string
        coords = site.frac_coords
        siteid = coord2idx(coords[0], coords[1], coords[2], unit_site, spcDesign)
        binvec[element_list.index(element), siteid] = 1
    if flatten:
        binvec = binvec.flatten()
    return binvec, scaling_matrix


def binvec2poscar(binvec, poscar, elements, alat, nsites, spc_size):
    system = {}
    system['comment'] = "binary vector to poscar structure by QALO"
    system['scale_coeff'] = float(1.0)
    system['box_coord'] = []
    system['box_coord'].append([spc_size[0] * alat, 0.0, 0.0])
    system['box_coord'].append([0.0, spc_size[1] * alat, 0.0])
    system['box_coord'].append([0.0, 0.0, spc_size[2] * alat])
    system['atom'] = elements

    compositions = []
    for i in range(len(elements)):
        compositions.append(int(sum(binvec[i * nsites: (i + 1) * nsites])))

    system['atom_num'] = compositions
    system['all_atom'] = sum(system['atom_num'])
    system['coord_type'] = 'Direct'
    system['coord_Direct'] = []
    indices = np.where(binvec == 1)[0]
    for a in range(int(system['all_atom'])):
        i, j = indices[a] // nsites, indices[a] % nsites
        x, y, z = idx2coord(j)
        system['coord_Direct'].append([x, y, z])

    with open(poscar, 'w') as f:
        f.writelines(" " + system['comment'] + "\n")
        f.writelines("   " + str(system['scale_coeff']) + "\n")
        for i in range(3):
            f.writelines("     " + ' '.join(str('%.16f' % x) for x in system['box_coord'][i]) + "\n")
        f.writelines(" " + ' '.join(str(x) for x in system['atom']) + "\n")
        f.writelines("  " + ' '.join(str(x) for x in system['atom_num']) + "\n")
        f.writelines(system['coord_type'] + "\n")
        for i in range(int(system['all_atom'])):
            if system['coord_type'] == 'Direct':
                f.writelines("   " + ' '.join(str('%.16f' % x) for x in system['coord_Direct'][i]) + "\n")
    return system


def raw2libffm(filepath, spcOriginal, spcDesign, unit_site, nsites):
    entries = os.listdir(filepath)
    random.shuffle(entries)
    libffm_data = []
    for structure in entries:
        d = os.path.join(filepath, structure)
        if os.path.isdir(d) and structure.startswith("s-"):
            binvec, scaling_matrix = poscar2binvec(poscar=os.path.join(d, "POSCAR"), 
                                                   spcOriginal=spcOriginal, 
                                                   spcDesign=spcDesign, 
                                                   unit_site=unit_site, 
                                                   flatten=True)
            toten = (float(extract_toten(os.path.join(d, "OUTCAR"))) * scaling_matrix[0][0] * scaling_matrix[1][1] * scaling_matrix[2][2])
            libffm_row = [str(toten)]
            for i, value in enumerate(binvec):
                field = i % nsites
                feature = (i // nsites) * nsites + field
                libffm_row.append(f"{field}:{feature}:{int(value)}")
            libffm_data.append(' '.join(libffm_row))
    return libffm_data
