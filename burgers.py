import finat
from firedrake import *
from firedrake.adjoint import *


continue_annotation()

tape = get_working_tape()
total_steps = 1000
num_sources = 1
source_number = 0
Lx, Lz = 1.0, 1.0
mesh = UnitSquareMesh(200, 200)
V = FunctionSpace(mesh, "CG", 1)

def Dt(u, u_, timestep):
    return (u - u_)/timestep


def test_memory_burger():
    u_ = Function(V)
    u = Function(V)
    v = TestFunction(V)
    timestep = Constant(0.0000001)
    x,_ = SpatialCoordinate(mesh)
    ic = Function(V).interpolate(0.5 + 0.1*sin(2*pi*x))
    u_.assign(ic)
    nu = Constant(0.001)
    F = (Dt(u, u_, timestep)*v + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    t = 0.0
    problem = NonlinearVariationalProblem(F, u, bcs=bc)
    solver = NonlinearVariationalSolver(problem)
    t += float(timestep)
    for t in tape.timestepper(iter(range(total_steps))):
        # print("step = ", t, "no revolve")
        solver.solve()
        u_.assign(u)
    J = assemble(u*u*dx)
    J_hat = ReducedFunctional(J, Control(ic))
    with stop_annotating():
        for _ in range(5):
            J_hat(Function(V).interpolate(0.5 + 0.1*sin(2*pi*x)))
            J_hat.derivative()


test_memory_burger()
