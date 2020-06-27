import torch

class puckNET(nn.Modle):
    def __init__(self, D_in, H, D_out, HL_count):
        """
        In the constructor we initalise the inherited parent class and instantiate our neuralnetwork layers
        """
        super(puckNET, self).__init__()
        self.input_layer = torch.nn.Linear(D_in, H)
        self.hidden_layer = torch.nn.Linear(H, H)
        self.dropout_layer = torch.nn.Dropout(p=0.5, inplace=False)
        self.output_layer = torch.nn.Linear(H, D_out)
        self.activation = torch.nn.ReLU()
        
    def forward(self, x):
        """
        In the forward function we accept a tensor and feed it forward through our neuralNet
        """
        h_relu = self.input_layer(x).clamp(min=0)
        h_relu = self.activation(h_relu)
        for _ in range(self.HL_count):
            h_relu = self.hidden_layer(h_relu).clamp(min=0)
            h_relu = self.activation(h_relu)
        y_pred = self.output_layer(h_relu)
        return torch.nn.tanh(y_pred, dim=1)

"""
    D_in = 
    H = 
    D_out = 2
    HL_count = 4    

    output activation function from -1 to 1


    stats of model
        hits
        shots
        saves
        goals

        if shots >> saves 
            agressive species

        if saves >> shots
            defensive species

        if shots ~= saves
            all rounder species

        if high hits
            if high shots
                accurate
            if low shots
                inacurate

    out of 40 new models
        brand new
        50% best scorer - 50% second best scorer
        


       


"""