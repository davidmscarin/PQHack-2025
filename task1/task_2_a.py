from qadence import Register, FeatureParameter, chain
from qadence import hea, AnalogRX, AnalogRY, AnalogRZ, AnalogInteraction, QuantumModel, QuantumCircuit, kron, RX, RY, RZ, Interaction, Z, add, X
import torch
import numpy as np
from load import load_txt
import matplotlib.pyplot as plt

from qadence.parameters import VariationalParameter

from qadence.operations.primitive import Y

from qadence.states import normalize

# estimate the value used to generate the data in data/dataset_1_a.txt
def estimate_value():
    # Load the data
    x_train, y_train = load_txt("../data/dataset_2_a.txt")

    x_train, y_train = torch.tensor(x_train), torch.tensor(y_train)

    # circuit with 1 qubit
    n_qubits = 2

    fm = kron(RX(0, FeatureParameter("x")), RX(1, 2*FeatureParameter("x")))

    # create ansazts
    ansatz = kron(RX(0, VariationalParameter("phi1")), RX(1, VariationalParameter("phi2")))
    block = chain(fm, ansatz)

    # create circuit
    circuit = QuantumCircuit(n_qubits, block)
    obs = (add(VariationalParameter("A"+str(i))*Z(i) for i in range(n_qubits)))

    # create model
    model = QuantumModel(circuit, observable = obs)
    y_pred_initial = model.expectation({"x": x_train}).squeeze().detach()
    print(y_pred_initial)

    # training loop

    criterion = torch.nn.MSELoss()

    def loss_fn(x_train, y_train):
        output = model.expectation({"x": x_train}).squeeze()
        loss = criterion(output, y_train)
        return loss
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

    n_epochs = 100

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(x_train, y_train)
        loss.backward()
        optimizer.step()
    
    y_pred_final = model.expectation({"x": x_train}).squeeze().detach()

    phi1 = model.vparams["phi1"].item()
    phi2 = model.vparams["phi2"].item()

    plt.plot(x_train, y_pred_initial, label = "Initial prediction")
    plt.plot(x_train, y_pred_final, label = "Final prediction")
    plt.scatter(x_train, y_train, label = "Training points")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title("phi1 = " + str(phi1) + ", phi2 = " + str(phi2))
    plt.xlim((-1, 8))
    plt.ylim((-2, 2))
    plt.show()

    return phi1, phi2


print(estimate_value())

