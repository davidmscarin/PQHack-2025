from qadence import Register, FeatureParameter, chain
from qadence import hea, AnalogRX, AnalogRY, AnalogRZ, AnalogInteraction, QuantumModel, QuantumCircuit, kron, RX, RY, RZ, Interaction, Z, add, X
import torch
import numpy as np
from load import load_txt
import matplotlib.pyplot as plt

from qadence.parameters import VariationalParameter

from qadence.operations.primitive import Y

# estimate the value used to generate the data in data/dataset_1_a.txt
def estimate_value():
    # Load the data
    x_train, y_train = load_txt("../data/dataset_1_a.txt")

    x_train, y_train = torch.tensor(x_train), torch.tensor(y_train)

    # circuit with 1 qubit
    n_qubits = 1

    fm = RX(0, FeatureParameter("x"))

    # create ansazts
    ansatz = RX(0, VariationalParameter("phi"))
    block = chain(fm, ansatz)

    # create circuit
    circuit = QuantumCircuit(n_qubits, block)
    obs = add(Y(i) for i in range(n_qubits))

    # create model
    model = QuantumModel(circuit, observable = obs)
    y_pred_initial = model.expectation({"x": x_train}).squeeze().detach()

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

    # plt.plot(x_train, y_pred_initial, label = "Initial prediction")
    # plt.plot(x_train, y_pred_final, label = "Final prediction")
    # plt.scatter(x_train, y_train, label = "Training points")
    # plt.xlabel("x")
    # plt.ylabel("f(x)")
    # plt.legend()
    # plt.title("phi = " + str(model.vparams["phi"].item()))
    # plt.xlim((-1, 8))
    # plt.ylim((-2, 2))
    # plt.show()

    return print(f"phi: {model.vparams["phi"].item()}")


estimate_value()

