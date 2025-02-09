from qadence import Register, FeatureParameter, chain
from qadence import hea, AnalogRX, AnalogRY, AnalogRZ, AnalogInteraction, QuantumModel, QuantumCircuit, kron, RX, RY, RZ, Interaction, Z, add, X
import torch
import numpy as np
from load import load_txt
import matplotlib.pyplot as plt
import math

from qadence.parameters import VariationalParameter

from qadence.operations.primitive import Y

from qadence.states import normalize

# estimate the value used to generate the data in data/dataset_1_a.txt
def estimate_value():
    # Euler method
    t_min = 0
    t_max = 10
    delta = 0.1
    
    k = 2.3
    x_0 = 1.0
    v_0 = 1.2
    n_steps = int((t_max-t_min)/delta)
    t_values = torch.linspace(t_min, t_max, n_steps)
    x_values = [x_0]
    v_values = [v_0]
    for i in range(n_steps-1):
        v_values.append(v_values[-1] - k*x_values[-1]*delta)
        x_values.append(x_values[-1] + v_values[-1]*delta)

    x_values = torch.tensor(x_values)

    # circuit with 1 qubit
    n_qubits = 1

    fm = kron(RX(0, math.sqrt(k) * FeatureParameter("x")))

    # create ansazts
    ansatz = chain(RX(0, VariationalParameter("phi")))
    block = chain(fm, ansatz)

    # create circuit
    circuit = QuantumCircuit(n_qubits, block)
    obs = sum(VariationalParameter("A"+str(i))*Z(i) for i in range(n_qubits))

    # create model
    model = QuantumModel(circuit, observable = obs)
    x_pred_initial = model.expectation({"x": t_values}).squeeze().detach()

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
        loss = loss_fn(t_values, x_values)
        loss.backward()
        optimizer.step()
    
    x_pred_final = model.expectation({"x": t_values}).squeeze().detach()

    # plt.plot(t_values, x_pred_initial, label = "Initial prediction")
    # plt.plot(t_values, x_pred_final, label = "Final prediction")
    # plt.scatter(t_values, x_values, label = "Training points")
    # plt.xlabel("x")
    # plt.ylabel("f(x)")
    # plt.legend()
    # plt.title("phi1 = ")
    # plt.xlim((-1, 15))
    # plt.ylim((-5, 5))
    # plt.show()

    test_data = np.loadtxt("../data/dataset_3_test.txt")
    #get predicted values on test data
    x_pred_test = model.expectation({"x": torch.tensor(test_data)}).squeeze().detach()
    #create csv with test, predicted values
    np.savetxt("solution_3_a.csv", np.column_stack((test_data, x_pred_test)), delimiter=",", header="x, f(x)", comments="", fmt="%f")


estimate_value()

