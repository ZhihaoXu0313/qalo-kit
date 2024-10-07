from qalo.arguments import arguments
from qalo.module.annealer import *
from qalo.module.fm import *
from qalo.module.mlp import *
from qalo.utils import save_energy, raw2libffm, redirect

import random
import pandas as pd
import xlearn as xl
from tqdm import tqdm


class qalorun(arguments):
    def __init__(self):
        super().__init__()
        self.Nx, self.Ny, self.Nz = self.spc_size[0], self.spc_size[1], self.spc_size[2]
        self.nspecies = len(self.elements)
        self.nsites = int(len(self.unit_site)) * self.Nx * self.Ny * self.Nz
        self.init_composition = [int(len(self.unit_site) * self.spc_size[0] * self.spc_size[1] * self.spc_size[2] * r) for r in self.init_composition_ratio]
        self.composition = self.init_composition
        
        self.fm_params = {'task': 'reg', 
                          'lr': self.fm_learning_rate, 
                          'lambda': self.fm_reg_lambda, 
                          'k': self.fm_latent_space, 
                          'opt': self.fm_opt, 
                          'epoch': self.fm_epoch, 
                          'metric': self.fm_metric}
        
    def write_libffm_data(self, datapath):
        """
        write libffm data from DFT dataset. data-axbxc --> libffm
        """
        datasets = os.listdir(datapath)
        libffm_data = []
        for ds in datasets:
            dims = ds.split('-')[-1]
            dimxyz = dims.split('x')
            spcOriginal = [int(dimxyz[0]), int(dimxyz[1]), int(dimxyz[2])]
            d = os.path.join(datapath, ds)
            libffm_data.extend(raw2libffm(filepath=d, spcOriginal=spcOriginal, spcDesign=self.spc_size, unit_site=self.unit_site, nsites=self.nsites))
        random.shuffle(libffm_data)
        with open(os.path.join(self.fm_directory, "libffm_data_total.txt"), 'a') as file:
            for line in libffm_data:
                file.write(line + '\n')
        return libffm_data
    
    def add_new_libffm_data(self, energy, dfstack):
        """
        add new structure-energy data to dataset & new structure list.
        """
        dfstack.insert(loc=0, column='energy', value=energy)
        new_libffm_data = []
        new_binvec_structure = []
        for index, row in dfstack.iterrows():
            label = row['energy']
            features = row.drop('energy')
            libffm_row = [str(label)]
            binvec_row = []
            for i, value in enumerate(features):
                field = i % self.nsites
                feature = (i // self.nsites) * self.nsites + field
                libffm_row.append(f"{field}:{feature}:{int(value)}")
                binvec_row.append(str(int(value)))
            new_libffm_data.append(' '.join(libffm_row))
            new_binvec_structure.append(' '.join(binvec_row))
        with open(os.path.join(self.fm_directory, "libffm_data_total.txt"), 'a') as file:
            for line in new_libffm_data:
                file.write(line + '\n')
        with open(os.path.join(self.new_data_directory, "new_structure.txt"), 'a') as file:
            for line in new_binvec_structure:
                file.write(line + '\n')
        return new_libffm_data, new_binvec_structure
    
    def annealing(self):
        energy_solution = []
        structuredf = pd.DataFrame()
        init_k = self.qa_relax * self.qa_constr
        growth_rate = self.qa_relax
        for i in range(self.qa_mix_circle):
            k = growth_rate ** (1 - i / self.qa_mix_circle)
            annealer_loose = annealer(nspecies=self.nspecies, 
                                      nsites=self.nsites, 
                                      temperature=self.temperature,
                                      placeholder=self.qa_constr, 
                                      fmpath=self.fm_directory, 
                                      composition=self.composition, 
                                      annealer_type=self.qa_type, 
                                      mode="loose", 
                                      ks=k)
            annealer_loose.run(n_sim=self.qa_shots, unit_sites=self.unit_site, spc_size=self.spc_size)
            sdfl = annealer_loose.extract_solutions() 

            energy = []
            energy_min = 0
            for n, s in sdfl.iterrows():
                e = snap_model_inference(binvec=np.array(s.values), 
                                         infile=os.path.join(self.input_directory, self.lmps_infile), 
                                         elements=self.elements, 
                                         alat=self.alat, 
                                         nsites=self.nsites, 
                                         spc_size=self.spc_size, 
                                         unit_site=self.unit_site,
                                         coeffile=os.path.join(self.input_directory, self.lmps_coeffile), 
                                         path_of_tmp=self.tmp_directory)
                energy.append(e)
                energy_min = min(energy_min, e)
            
            if i != self.qa_mix_circle - 1:
                min_index = energy.index(energy_min)
                min_structure = sdfl.iloc[min_index, :].tolist()
                for j in range(self.nspecies):
                    self.composition[j] = sum(min_structure[j * self.nsites: (j + 1) * self.nsites])
            else:
                energy_solution = energy
                structuredf = sdfl      
        return energy_solution, structuredf
    
    def labeling(self, energy_solution, structuredf):
        self.add_new_libffm_data(energy=energy_solution, dfstack=structuredf)
        csvfile = "mix-energy-composition.csv"
        save_energy(csvpath=os.path.join(self.output_directory, csvfile), energy=energy_solution, composition=self.composition)
        return self.composition
    
    def fmtraining(self, ffmModel):
        split_libffm_data(fmpath=self.fm_directory, sample_ratio=self.fm_sampling_ratio)
        with redirect("xlearn.log"):
            fm_train(model=ffmModel, 
                     param=self.fm_params, 
                     trainSet=os.path.join(self.fm_directory, "train_ffm.txt"),
                     validSet=os.path.join(self.fm_directory, "valid_ffm.txt"), 
                     model_txt=os.path.join(self.fm_directory, "model.txt"),
                     model_out=os.path.join(self.fm_directory, "train.model"),
                     restart=True)
            
    def run(self):
        print("Start the optimization...")
        print("Setup working folders...")
        if not os.path.exists(self.tmp_directory):
            os.mkdir(self.tmp_directory)
        if not os.path.exists(self.lmps_directory):
            os.mkdir(self.lmps_directory)
        if not os.path.exists(self.mlp_directory):
            os.mkdir(self.mlp_directory)
        if not os.path.exists(self.fm_directory):
            os.mkdir(self.fm_directory)
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)
        print("Finished")    
        
        print("Step 1: Generating initial libffm data")
        libffm_data = self.write_libffm_data(datapath=self.dft_data_directory)
        print("Step 2: Starting iterations")
        print(f"Total QALO iterations: {self.iterations}")
        print(f"Quantum annealing shots: {self.qa_shots}")
        
        ffmModel = xl.create_ffm()
        for i in tqdm(range(self.iterations)):
            self.fmtraining(ffmModel)
            energy_solution, structuredf = self.annealing()
            self.labeling(energy_solution, structuredf)
        print("Finished")
        
        