from qadence import QuantumCircuit, QuantumModel, FeatureParameter, HamEvo
from qadence import RX, RY, VariationalParameter, Z, add, hea, chain, kron, Y
from qadence import (FeatureParameter, RX, Z, hea, chain,
                    hamiltonian_factory, QuantumCircuit,
                    QuantumModel, BackendName, DiffMode)
import torch
import numpy as np
import math
import matplotlib.pyplot as plt

def solution(t):
    if t == 0:
        return 1
    return np.exp(k * t)

# Define parameters
k = -2.5
t_vals = np.linspace(0, 1, 100)
t_vals = {"x": torch.tensor(t_vals)}
x_initial = 1
n_epochs = 100
learning_rate = 0.1
phi_value = 0.5

from scipy.optimize import minimize

# quantum model
n_qubits = 1
depth = 2
fm = kron(RX(i, FeatureParameter("t")) for i in range(n_qubits))
ansatz = RX(0, VariationalParameter("phi"))
block = chain(fm, ansatz)
circuit = QuantumCircuit(n_qubits, block)
obs = add(Z(i) for i in range(n_qubits))
model = QuantumModel(circuit, observable = obs)

# model pred
def f_model(t):
    return model.expectation({"t": torch.tensor([t])}).item()

#compute derivative according to psr
def derivative(t):
    return 1/2*(model.expectation({"t": torch.tensor([t+np.pi/2])}) - model.expectation({"t": torch.tensor([t-np.pi/2])}))


def exact(t):
    return np.exp(k*t)

#training loop
for epoch in range(n_epochs):
    loss = 0
    for t in np.linspace(0, 1, 100):  # Training points in [0,1]
        f_t = f_model(t)
        dfdt_t = derivative(t)
        loss += (dfdt_t - k * f_t) ** 2  # Residual loss

    # Gradient descent update
    grad_phi = -2 * (derivative(0) - k * f_model(0))  # Compute gradient
    phi_value -= learning_rate * grad_phi
    model.vparams["phi"] = phi_value  # Update variational parameter

# # Plot the final solution
# t_vals = np.linspace(0, 1, 100)
# x_vals = [f_model(t) for t in t_vals]
# solutions = [exact(t) for t in t_vals]

# plt.plot(t_vals, x_vals, label="Quantum Model Approximation")
# plt.plot(t_vals, solutions, "--", label="Exact Solution $e^{kt}$")
# plt.xlabel("t")
# plt.ylabel("x(t)")
# plt.legend()
# plt.show()

#create csv with t, predicted values
test_data = np.loadtxt("../data/dataset_5_test.txt")
# get predicted values on test data
x_pred_test = model.expectation({"t": torch.tensor(test_data)}).squeeze().detach()
np.savetxt("solution_5.csv", np.column_stack((test_data, x_pred_test)), delimiter=",", header="t, f(t)", comments="", fmt="%f")


