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
    x_train, y_train = load_txt("../data/dataset_2_c.txt")

    x_train, y_train = torch.tensor(x_train), torch.tensor(y_train)

    # circuit with 1 qubit
    n_qubits = 3

    fm = kron(
        RX(0, FeatureParameter("x")**0),
        RX(1, FeatureParameter("x")**1),
        RX(2, FeatureParameter("x")**2)
    )

    # create ansazts
    ansatz = chain(hea(3, 3))
    block = chain(fm, ansatz)

    # create circuit
    circuit = QuantumCircuit(n_qubits, block)
    obs = add(add(VariationalParameter("A"+str(i))*Y(i) for i in range(n_qubits)), add(VariationalParameter("B"+str(i))*Z(i) for i in range(n_qubits)))

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
    
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)

    n_epochs = 100

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model, x_train, y_train)
        loss.backward()
        optimizer.step()

    y_pred_final = model.expectation({"x": x_train}).squeeze().detach()

    # plt.plot(x_train, y_pred_initial, label = "Initial prediction")
    # plt.plot(x_train, y_pred_final, label = "Final prediction")
    # plt.scatter(x_train, y_train, label = "Training points")
    # plt.xlabel("x")
    # plt.ylabel("f(x)")
    # plt.legend()
    # plt.title("phi1 = ")
    # plt.xlim((-1, 8))
    # plt.ylim((-2, 2))
    # plt.show()

    #create csv with test, prediction
    data = np.loadtxt("../data/dataset_2_c_test.txt")
    x_test = data[:, 0]
    y_test = data[:, 1]
    pred = model.expectation({"x": torch.tensor(x_test)}).squeeze().detach()
    np.savetxt("solution_2_c.csv", pred.numpy(), delimiter=",")

estimate_value()

