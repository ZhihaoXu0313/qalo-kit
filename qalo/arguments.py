import os


class arguments:
    def __init__(self):
        # basic
        self.iterations = 100
        self.elements = ["Nb", "Mo", "Ta", "W"]
        self.init_composition_ratio = [0.25, 0.25, 0.25, 0.25]
        self.spc_size = [4, 4, 4]
        self.unit_site = [[0, 0, 0], [0.5, 0.5, 0.5]]
        self.alat = 3.29
        self.temperature = 300
        self.init_composition = [int(len(self.unit_site) * self.spc_size[0] * self.spc_size[1] * self.spc_size[2] * r) for r in self.init_composition_ratio]
        
        # work directory
        self.working_directory = os.getcwd()
        self.input_directory = os.path.join(self.working_directory, "input")
        self.output_directory = os.path.join(self.working_directory, "output")
        self.tmp_directory = os.path.join(self.working_directory, "tmp")
        self.fm_directory = os.path.join(self.working_directory, "fm")
        self.mlp_directory = os.path.join(self.working_directory, "mlp")
        self.lmps_directory = os.path.join(self.working_directory, "lmps")
        self.dft_data_directory = os.path.join(self.working_directory, "database", "dft")
        self.new_data_directory = os.path.join(self.working_directory, "database", "new")
        self.lmps_infile = "in.snap.lmp"
        self.lmps_coeffile = "NbMoTaW.snapcoeff"
        
        # factorization machine
        self.fm_learning_rate = 0.05
        self.fm_reg_lambda = 0.05
        self.fm_latent_space = 8
        self.fm_epoch = 10000
        self.fm_metric = "rmse"
        self.fm_opt = "adagrad"
        self.fm_sampling_ratio = 0.7
        
        # quantum annealer
        self.qa_type = "qasim"
        self.qa_constr = 5e+2
        self.qa_relax = 1e-2
        self.qa_shots = 5
        self.qa_mix_circle = 5
        
    def show(self):
        print("==========")
        print("Parameters")
        for key, value in self.__dict__.items():
            print(f'{key}: {value}')
        print("==========")
        

if __name__ == "__main__":
    args = arguments()
    args.show()
    