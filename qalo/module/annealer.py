import numpy as np 
import re 
import pandas as pd 
import os 
from pyqubo import Array, Binary, Placeholder, Constraint
import neal
from dwave.system import LeapHybridSampler, DWaveSampler, FixedEmbeddingComposite, FixedEmbeddingComposite
import dimod
from dimod import BinaryQuadraticModel
from dwave_qbsolv import QBSolv 
import math

import networkx as nx 
import minorminer

from qalo.utils import idx2coord, coord2idx, stack_dataframe
from qalo.module.fm import load_fm_model


def translate_result(df, unit_sites, spc_size):
    L = df.columns[df.values[0] == 1]
    table = np.zeros((len(L), 4))
    for m in range(len(L)):
        i, j = extract_idx(df.columns[df.values[0] == 1][m])
        x, y, z = idx2coord(idx=j, site_positions=unit_sites, supercell=spc_size)
        table[m, 0] = i
        table[m, 1] = x
        table[m, 2] = y
        table[m, 3] = z
    return table


def extract_idx(s):
    pattern = r'x\[(\d+)\]\[(\d+)\]'
    match = re.search(pattern, s)
    
    if match:
        i = int(match.group(1))
        j = int(match.group(2))
        return i, j
    else:
        print("pattern not found")
        return 1, -1
    

def table_to_map(table, spc_size, unit_site, elements):
    Nx, Ny, Nz = spc_size[0], spc_size[1], spc_size[2]
    nsites = int(len(unit_site)) * Nx * Ny * Nz
    
    data = np.zeros((len(elements), nsites))
    for i in range(len(table)):
        idx = coord2idx(x=table[i, 1], y=table[i, 2], z=table[i, 3], site_positions=unit_site, supercell=spc_size)
        data[int(table[i, 0]), idx] = 1
    return data


def simulate_annealing(bqm):
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm)
    return sampleset


def qbsolv_annealing(Q, subQuboSize=45):
    G = nx.complete_graph(subQuboSize)
    system = DWaveSampler()
    embedding = minorminer.find_embedding(G.edge, system.edgelist)
    sampler = FixedEmbeddingComposite(system, embedding)
    sampleset = QBSolv.sample_qubo(Q, solver=sampler, solver_limit=subQuboSize, label='QUBO Optimization')
    return sampleset


def hybrid_quantum_annealing(bqm):
    sampler = LeapHybridSampler()
    sampleset = sampler.sample(bqm)
    return sampleset


class hamiltonian:
    def __init__(self, nspecies, nsites):
        self.nspecies = nspecies
        self.nsites = nsites
        self.x = Array.create('x', shape=(self.nspecies, self.nsites), vartype='BINARY')
        self.M = Placeholder('M')
        self.H = 0
        
    def construct_hamiltonian(self, model_txt):
        Q, offset = load_fm_model(model_txt, self.nspecies * self.nsites, self.nsites)
        for i in range(self.nspecies):
            for j in range(self.nsites):
                for k in range(self.nspecies):
                    for l in range(self.nsites):
                        self.H += Q[i * self.nsites + j, k * self.nsites + l] * self.x[i, j] * self.x[k, l]
                        
    def add_entropy(self, temperature):
        R = 5.189e19 * self.nsites / 6.02e23 # J/K/mol * total_mol --> eV/K
        entropy_conf = 0
        
        for i in range(self.nspecies):
            frac_composition = sum(self.x[i, :]) / self.nsites
            entropy_conf -= R * frac_composition * math.log(frac_composition)
        self.H -= temperature * entropy_conf
    
    def apply_constraints(self, composition, mode, scale):
        K1 = self.M if mode == 'tight' else self.M * scale
        for i in range(self.nspecies):
            self.H += K1 * (sum(self.x[i, :]) - composition[i]) ** 2
            
        K2 = self.M
        for j in range(self.nsites):
            for i in range(self.nspecies):
                self.H += K2 * self.x[i, j] * (sum(self.x[:, j]) - 1)
                
    def compile_hamiltonian(self):
        return self.H.compile()
    
    def translate(self, coeff, obj):
        model = self.compile_hamiltonian()
        if obj == 'bqm':
            bqm = model.to_bqm(feed_dict={'M': coeff})
            return bqm
        elif obj == 'qubo':
            qubo, offset = model.to_qubo(feed_dict={'M': coeff})
            return qubo, offset
        elif obj == 'ising':
            ising, offset = model.to_ising(feed_dict={'M': coeff})
            return ising, offset
        else:
            print("Invalid translated format!!!")
            

class annealer:
    def __init__(self, nspecies, nsites, temperature, placeholder, fmpath, composition, annealer_type, mode, ks):
        self.nspecies = nspecies
        self.nsites = nsites
        self.placeholder = placeholder
        
        self.hQubo = hamiltonian(nspecies, nsites)
        self.hQubo.construct_hamiltonian(os.path.join(fmpath, "model.txt"))
        self.hQubo.add_entropy(temperature=temperature)
        self.hQubo.apply_constraints(composition, mode, ks)
        self.hQubo.compile_hamiltonian()
        
        self.bqm = self.hQubo.translate(placeholder, 'bqm')
        self.Q, self.offset = self.hQubo.translate(placeholder, 'qubo')
        
        self.annealer_type = annealer_type
        
        self.structuredfstack = pd.DataFrame()
    
    def run(self, n_sim, unit_sites, spc_size):
        if self.annealer_type == 'qasim':
            for _ in range(n_sim):
                sampleset = simulate_annealing(self.bqm)
                dfs = pd.DataFrame(sampleset.lowest())
                self.structuredfstack = stack_dataframe(dfs, self.structuredfstack)
                result = translate_result(df=dfs, unit_sites=unit_sites, spc_size=spc_size)
        elif self.annealer_type == 'hybrid':
            for _ in range(n_sim):
                sampleset = hybrid_quantum_annealing(self.bqm)
                dfs = pd.DataFrame(sampleset.lowest())
                self.structuredfstack = stack_dataframe(dfs, self.structuredfstack)
                result = translate_result(df=dfs, unit_sites=unit_sites, spc_size=spc_size)
                
    def extract_solutions(self):
        columns_sorted = sorted(self.structuredfstack.columns, key=extract_idx)
        return self.structuredfstack[columns_sorted]
