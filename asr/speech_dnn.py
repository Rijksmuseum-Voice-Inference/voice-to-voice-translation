class MLP(nn.Module):
    """docstring for MLP"""
    def __init__(self, 
                input_dim, 
                output_dim, 
                n_hidden, 
                width,
                activation=nn.ReLU(),
                BN=False):
        super(MLP, self).__init__()
        
        if n_hidden == 0:
            width = input_dim

        modules = []
        for i in range(n_hidden):
            if i == 0:
                modules.append(nn.Linear(input_dim, width))
            else:
                modules.append(nn.Linear(width, width))
            if BN:
                modules.append(nn.BatchNorm1d(width))
        modules.append(nn.Linear(width, output_dim))
        self.linears = nn.ModuleList(modules)
        self.activation = activation

    def forward(self, x):
        for l in self.linears:
            x = l(x)
            x = self.activation(x)
        return x

def load_model(input_dim, output_dim, state_file_path):
    model = MLP(input_dim, output_dim, 3, 2048, BN=True)
    params = torch.load(state_file_path)
    model.load_state_dict(params['state'])