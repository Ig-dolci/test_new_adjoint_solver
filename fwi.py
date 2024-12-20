from firedrake import *
M = 1
my_ensemble = Ensemble(COMM_WORLD, M)

num_sources = my_ensemble.ensemble_comm.size

# The source number is defined with the ``Ensemble.ensemble_comm`` rank::

source_number = my_ensemble.ensemble_comm.rank
  
Lx, Lz = 1.0, 1.0
mesh = UnitSquareMesh(200, 200, comm=my_ensemble.comm)

# The basic input for the FWI problem are defined as follows::

import numpy as np
source_locations = np.linspace((0.3, 0.1), (0.7, 0.1), num_sources)
receiver_locations = np.linspace((0.2, 0.9), (0.8, 0.9), 20)
dt = 0.0005  # time step in seconds
final_time = 0.8  # final time in seconds
frequency_peak = 7.0  # The dominant frequency of the Ricker wavelet in Hz.

V = FunctionSpace(mesh, "KMV", 1)
x, z = SpatialCoordinate(mesh)
c_true = Function(V).interpolate(1.75 + 0.25 * tanh(200 * (0.125 - sqrt((x - 0.5) ** 2 + (z - 0.5) ** 2))))


def ricker_wavelet(t, fs, amp=1.0):
    ts = 1.5
    t0 = t - ts * np.sqrt(6.0) / (np.pi * fs)
    return (amp * (1.0 - (1.0 / 2.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0)
            * np.exp((-1.0 / 4.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0))

source_mesh = VertexOnlyMesh(mesh, [source_locations[source_number]])
V_s = FunctionSpace(source_mesh, "DG", 0)
d_s = Function(V_s)
d_s.interpolate(1.0)
source_cofunction = assemble(d_s * TestFunction(V_s) * dx)
q_s = Cofunction(V.dual()).interpolate(source_cofunction)

import finat

def wave_equation_solver(c, source_function, dt, V):
    u = TrialFunction(V)
    v = TestFunction(V)
    u_np1 = Function(V)  # timestep n+1
    u_n = Function(V)  # timestep n
    u_nm1 = Function(V)  # timestep n-1
    # Quadrature rule for lumped mass matrix.
    quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")
    m = (1 / (c * c))
    time_term =  m * (u - 2.0 * u_n + u_nm1) / Constant(dt**2) * v * dx(scheme=quad_rule)
    nf = (1 / c) * ((u_n - u_nm1) / dt) * v * ds
    a = dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)
    F = time_term + a + nf
    lin_var = LinearVariationalProblem(lhs(F), rhs(F) + source_function, u_np1, constant_jacobian=True)
    # Since the linear system matrix is diagonal, the solver parameters are set to construct a solver,
    # which applies a single step of Jacobi preconditioning.
    solver_parameters = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
    solver = LinearVariationalSolver(lin_var,solver_parameters=solver_parameters)
    return solver, u_np1, u_n, u_nm1


receiver_mesh = VertexOnlyMesh(mesh, receiver_locations)
V_r = FunctionSpace(receiver_mesh, "DG", 0)

from firedrake.__future__ import interpolate

true_data_receivers = []
total_steps = int(final_time / dt) + 1
f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = wave_equation_solver(c_true, f, dt, V)
interpolate_receivers = interpolate(u_np1, V_r)

for step in range(total_steps):
    f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    true_data_receivers.append(assemble(interpolate_receivers))

c_guess = Function(V).interpolate(1.5)

from firedrake.adjoint import *
continue_annotation()

# **Steps 2-3**: Solve the wave equation and compute the functional::
tape = get_working_tape()
tape.progress_bar = ProgressBar
f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = wave_equation_solver(c_guess, f, dt, V)
interpolate_receivers = interpolate(u_np1, V_r)
J_val = 0.0
for step in range(total_steps):
    f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    guess_receiver = assemble(interpolate_receivers)
    misfit = guess_receiver - true_data_receivers[step]
    J_val += 0.5 * assemble(inner(misfit, misfit) * dx)


J_hat = EnsembleReducedFunctional(J_val, Control(c_guess), my_ensemble)


c_optimised = minimize(J_hat, method="L-BFGS-B", options={"disp": True, "maxiter": 5},
                       bounds=(1.5, 2.0), derivative_options={"riesz_representation": 'l2'}
                       )
