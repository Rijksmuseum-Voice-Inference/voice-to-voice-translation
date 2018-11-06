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

        activation = Activation(activation)

        modules = []
        for i in range(n_hidden):
            if i == 0:
                modules.append(nn.Linear(input_dim, width))
            else:
                modules.append(nn.Linear(width, width))
            if BN:
                modules.append(nn.BatchNorm1d(width))
            modules.append(activation)

        if n_hidden == 0:
            width = input_dim

        self.hidden = nn.ModuleList(modules)
        self.output = nn.Linear(width, output_dim)
        self.start = 0
        self.end = len(self.hidden)

    def setForwardRange(self, start, end):
        assert(start < end)
        assert(end <= len(self.hidden))
        self.start = start
        self.end = end

    def resetForwardRange(self, start, end):
        self.start = 0
        self.end = len(self.hidden)

    def fineTuneLayer(self, i):
        assert(i < len(self.hidden))
        for j in range(len(self.hidden)):
            for p in self.hidden[j].parameters():
                    p.requires_grad = i != j

    def trainAllLayers(self):
        for j in range(len(self.hidden)):
            for p in self.hidden[j].parameters():
                    p.requires_grad = True

    def forward(self, x):
        for i in range(self.start, self.end):
            x = self.hidden[i](x)
        x = self.output(x)
        return x

def load_model(input_dim, output_dim, state_file_path):
    model = MLP(input_dim, output_dim, 3, 2048, BN=True)
    params = torch.load(state_file_path)
    model.load_state_dict(params['state'])