from qalo.arguments import arguments


class qalorun(arguments):
    def __init__(self):
        super().__init__()
        self.Nx, self.Ny, self.Nz = self.spc_size[0], self.spc_size[1], self.spc_size[2]
        self.nspecies = len(self.elements)
        self.nsites = int(len(self.unit_site)) * self.Nx * self.Ny * self.Nz
        
    def annealer(self, ):
        pass
    
    def labeling(self):
        pass
    
    def fmtrain(self):
        pass
