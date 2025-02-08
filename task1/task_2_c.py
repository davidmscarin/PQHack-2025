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
    x_train, y_train = load_txt("../data/dataset_2_c.txt")

    x_train, y_train = torch.tensor(x_train), torch.tensor(y_train)

    # circuit with 1 qubit
    n_qubits = 3

    fm = kron(RX(i, VariationalParameter("f"+str(i+1))*FeatureParameter("x")) for i in range(n_qubits))

    # create ansazts
    ansatz = kron(RX(i, VariationalParameter("phi"+str(i+1))) for i in range(n_qubits))
    block = chain(fm, ansatz)

    # create circuit
    circuit = QuantumCircuit(n_qubits, block)
    obs = add(VariationalParameter("A"+str(i+1))*Z(i) for i in range(n_qubits))

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

    plt.plot(x_train, y_pred_initial, label = "Initial prediction")
    plt.plot(x_train, y_pred_final, label = "Final prediction")
    plt.scatter(x_train, y_train, label = "Training points")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title("phi")
    plt.xlim((-1, 30))
    plt.ylim((-5, 5))
    plt.show()

    x_test, y_test = load_txt("../data/dataset_2_c_test.txt")

    x_test, y_test = torch.tensor(x_test), torch.tensor(y_test)

    y_pred_test = model.expectation({"x": x_test}).squeeze().detach()

    plt.plot(x_test, y_pred_test, label = "Final prediction")
    plt.scatter(x_test, y_test, label = "Training points")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title("phi")
    plt.xlim((-1, 30))
    plt.ylim((-5, 5))
    plt.show()

    print(model.vparams)


print(estimate_value())

