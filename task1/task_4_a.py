from qadence import Register, FeatureParameter, chain
from qadence import hea, AnalogRX, AnalogRY, AnalogRZ, AnalogInteraction, QuantumModel, QuantumCircuit, kron, RX, RY, RZ, Interaction, Z, add, X
import torch
import numpy as np
from load import load_txt
import matplotlib.pyplot as plt

from qadence.parameters import VariationalParameter

from qadence.operations.primitive import Y

from qadence.states import normalize

def evaluate(model, x, y):
    return model.expectation({"x": x, "y": y})

def pde_eval(model, delta, x, y, k, omega1, omega2):
    # Compute u(x, y)
    u_xy = evaluate(model, x, y)

    # Create a mask for boundary conditions
    boundary_mask = (x == -1) | (x == 1) | (y == -1) | (y == 1)

    # Compute finite difference approximations using broadcasting
    u_x_plus = evaluate(model, x + delta, y)
    u_x_minus = evaluate(model, x - delta, y)
    u_y_plus = evaluate(model, x, y + delta)
    u_y_minus = evaluate(model, x, y - delta)

    # Second-order partial derivatives
    partial_second_derivative_x = (u_x_plus - u_x_minus - 2 * u_xy) / delta**2
    partial_second_derivative_y = (u_y_plus - u_y_minus - 2 * u_xy) / delta**2

    # Compute the PDE residual
    pde_residual = partial_second_derivative_x + partial_second_derivative_y + ((k**2) * u_xy) - ((k**2 - omega1**2 - omega2**2) * torch.sin(omega1 * x) * torch.sin(omega2 * y)).reshape(-1, 1)

    # Apply boundary condition: enforce u(x, y) = 0 on boundaries
    pde_residual[boundary_mask] = u_xy[boundary_mask]

    return pde_residual

def evaluate_full(model):
    n_vals = 50

    x = torch.linspace(-1, 1, n_vals)
    y = torch.linspace(-1, 1, n_vals)

    x, y = torch.meshgrid(x, y, indexing='ij')

    results = evaluate(model, x.flatten(), y.flatten()).view(n_vals, n_vals)

    return results


def loss_fn(model, delta, k, omega1, omega2):
    n_vals = 50
    
    x = torch.linspace(-1, 1, n_vals)
    y = torch.linspace(-1, 1, n_vals)

    x, y = torch.meshgrid(x, y, indexing='ij')

    results = pde_eval(model, delta, x.flatten(), y.flatten(), k, omega1, omega2).view(n_vals, n_vals)

    total_loss = torch.sum(results**2)

    return total_loss/(n_vals**2)



# estimate the value used to generate the data in data/dataset_1_a.txt
def estimate_value():
    # Fix seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # circuit with 1 qubit
    n_qubits = 2
    depth = 3

    fm = kron(RX(0, torch.pi*FeatureParameter("x")), RX(1, 2*torch.pi*FeatureParameter("y")))

    # create ansazts
    ansatz = chain(hea(n_qubits, depth))
    block = chain(fm, ansatz)

    # create circuit
    circuit = QuantumCircuit(n_qubits, block)
    obs = 1/2*sum(Z(0) + Y(1))

    # create model
    model = QuantumModel(circuit, observable = obs)
    x_pred_initial = evaluate_full(model)

    x = torch.linspace(-1, 1, 50)
    y = torch.linspace(-1, 1, 50)
    x, y = torch.meshgrid(x, y)

    #plot the results
    plt.figure(figsize=(8, 6))
    plt.contourf(x, y, x_pred_initial.detach())

    plt.title(r'Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()

    # training loop
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

    n_epochs = 200

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model, 0.1, 1, np.pi, 2*np.pi)
        loss.backward()
        optimizer.step()
        print("Epoch: ", epoch, " Loss: ", loss.item())

    x_pred_final = evaluate_full(model)

    x = torch.linspace(-1, 1, 50)
    y = torch.linspace(-1, 1, 50)
    x, y = torch.meshgrid(x, y)

    #plot the results
    plt.figure(figsize=(8, 6))
    plt.contourf(x, y, x_pred_final.detach())

    plt.title(r'Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()

    test_data = np.loadtxt("../data/dataset_4_test.txt")
    x_test = test_data[:, 0]
    y_test = test_data[:, 1]
    #get predicted values on test data
    x_pred_test = model.expectation({"x": torch.tensor(x_test), "y": torch.tensor(y_test)}).squeeze().detach()
    #create csv with x, y, predicted values
    np.savetxt("solution_4.csv", np.column_stack((x_test, y_test, x_pred_test)), delimiter=",", header="x, y, f(x, y)", comments="", fmt="%f")

    return


print(estimate_value())