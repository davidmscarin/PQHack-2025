import math
from qadence import Register, FeatureParameter, chain
from qadence import hea, AnalogRX, AnalogRY, AnalogRZ, AnalogInteraction, QuantumModel, QuantumCircuit, kron, RX, RY, RZ, Interaction, Z, add, X
import torch
import numpy as np
from load import load_txt
import matplotlib.pyplot as plt
from sympy import floor
from copy import deepcopy

from qadence.parameters import VariationalParameter

from qadence.states import normalize

from qadence.operations.ham_evo import HamEvo

from qadence.operations.primitive import Y

# estimate the value used to generate the data in data/dataset_1_a.txt
def estimate_value():
    # fix seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed) 

    # Load the data
    x_train, y_train = load_txt("../data/dataset_2_b.txt")

    x_train, y_train = torch.tensor(x_train), torch.tensor(y_train)

    # circuit with 1 qubit
    n_qubits = 1
    n_shards = 3 # multiple sums of f*x and phi (meaning that the f given by the optimization will be n_shards times the actual f and the phi will be n_shards times the actual phi)
    fm = chain(chain(RX(0, VariationalParameter("f")*FeatureParameter("x")),
               RX(0, VariationalParameter("phi"))) for i in range(n_shards)
    )

    # create ansazts
    block = chain(fm)

    # create circuit
    circuit = QuantumCircuit(n_qubits, block)
    obs = VariationalParameter("A")*Z(0)

    # create model
    model = QuantumModel(circuit, observable = obs)
    y_pred_initial = model.expectation({"x": x_train}).squeeze().detach()
    print(y_pred_initial)

    # training loop

    criterion = torch.nn.MSELoss()

    def loss_fn(model, x_train, y_train):
        output = model.expectation({"x": x_train}).squeeze()
        loss = criterion(output, y_train)
        # print loss's computation graph (only for debugging)
        
        return loss
    
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.1)

    n_epochs = 100

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model, x_train, y_train)
        loss.backward()
        optimizer.step()

    y_pred_final = model.expectation({"x": x_train}).squeeze().detach()

    plt.plot(x_train, y_pred_initial, label = "Initial prediction")
    plt.plot(x_train, y_pred_final, label = "Final prediction")
    plt.scatter(x_train, y_train, label = "Training points")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title("phi1 = ")
    plt.xlim((-1, 8))
    plt.ylim((-2, 2))
    plt.show()

    f = model.vparams["f"].item()*n_shards
    phi = model.vparams["phi"].item()*n_shards
    A = model.vparams["A"].item()

    return f, phi, A


print(estimate_value())

