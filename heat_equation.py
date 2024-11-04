from firedrake import *
from firedrake.adjoint import *
continue_annotation()


def test_bdy_control():
    # Test for the case the boundary condition is a control for a
    # domain with length different from 1.
    mesh = SquareMesh(200, 200, 2.0, 2.0)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)
    trial = TrialFunction(space)
    sol = Function(space, name="sol")
    # Dirichlet boundary conditions
    R = FunctionSpace(mesh, "R", 0)
    a = Function(R, val=1.0)
    b = Function(R, val=2.0)
    bc_left = DirichletBC(space, a, 1)
    bc_right = DirichletBC(space, b, 2)
    bc = [bc_left, bc_right]
    u = TrialFunction(space)
    u_ = Function(space)
    time_step = Constant(0.001)
    time_term = (u - u_)/time_step * test * dx
    k = Function(space)
    k.interpolate(1.0)
    F = time_term + k * dot(grad(trial), grad(test)) * dx
    problem = LinearVariationalProblem(lhs(F), rhs(F), sol, bcs=bc, constant_jacobian=True)
    solver = LinearVariationalSolver(problem)
    for i in range(1000):
        print(f"Step {i}")
        solver.solve()
        u_.assign(sol)
    return assemble(0.5 * (sol * sol) * dx), k, space, R
    

J, k, space, R = test_bdy_control()
with stop_annotating():
    Jhat = ReducedFunctional(J, Control(k))
    Jhat.tape.progress_bar = ProgressBar
    for i in range(5):
        k.assign(1.0 + 0.1 * i)
        Jhat(k)
        Jhat.derivative()
