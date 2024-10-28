from firedrake import *
from firedrake.adjoint import *
import numpy as np
import matplotlib.pyplot as plt


class NS_simulation():
    def __init__(self):
        file = "/Users/ddolci/tes_fire_install/firedrake_petsc_new/test_new_adjoint_solver/"
        self.mesh = Mesh(file + "symmetric_mesh.msh")
        # coordinates = np.loadtxt("solution/Re_200/mesh_final.txt", delimiter=',')
        # self.mesh.coordinates.dat.data[:,:] = coordinates

        Vu = VectorFunctionSpace(self.mesh, "CG", 2)
        Vp = FunctionSpace(self.mesh, "CG", 1)
        self.V = MixedFunctionSpace([Vu, Vp])

        R = FunctionSpace(self.mesh, "R", 0)
        self.Re = Function(R)
        self.sol = Function(self.V)
        self.sol_test = TestFunction(self.V)
        self.pvd = File("NS_200/sol.pvd")

        self.solver_params = {
                                "mat_type": "aij",
                                "snes_monitor": None,
                                "snes_linesearch_type": "basic",
                                "snes_max_it": 100,
                                "snes_atol": 1.0e-9,
                                "snes_rtol": 0.0,
                                "ksp_type": "preonly",
                                "pc_type": "lu",
                                "pc_factor_mat_solver_type": "mumps"
                                }
        
        # Get Boundary conditions
        self.boundary_conditions()
        
    def residual(self, sol, sol_test):
        # Gaussian source term
        x, y = SpatialCoordinate(self.mesh)
        x_c, y_c = -1.5, 0.0
        (u, p) = split(sol)
        (v, q) = split(sol_test)
        self.g = exp(-100*((x - x_c)**2 + (y - y_c)**2))
        self.f = assemble(inner(as_vector([self.g, self.g]), v) * dx)
        F = -(
              (1/self.Re)*inner(grad(u), grad(v))*dx
              + inner(grad(u)*u, v)*dx
              - div(v)*p*dx
              + q*div(u)*dx
              - self.f
            )
        return F

    def boundary_conditions(self):
        self.bcs = [DirichletBC(self.V.sub(0), Constant((1,0)), (10,12)),
                    DirichletBC(self.V.sub(0), Constant((0,0)), (13))]

    def solve(self, Re):
        self.Re.assign(Re)
        continue_annotation()
        F = self.residual(self.sol, self.sol_test)
        solve(F == 0, self.sol, bcs=self.bcs, solver_parameters=self.solver_params)

    def compute_drag_lift(self):
        (u, p) = split(self.sol)
        sigma = 1/self.Re*(grad(u) + grad(u).T) - p * Identity(2)
        drag = -2 * assemble(dot(sigma, FacetNormal(self.mesh))[0]*ds(13))
        lift = 2 * assemble(dot(sigma, FacetNormal(self.mesh))[1]*ds(13))
        return drag, lift

    def save_solution(self):
        u = self.sol.split()[0]
        p = self.sol.split()[1]
        u.rename("u", "u")
        p.rename("p", "p")
        self.pvd.write(u, p)

    def transient(self):
        continue_annotation()
        # Delta_t = 1/12
        # Tfinal = 60.0
        Delta_t = 1/48
        Tfinal = 5.0
        List_t = np.arange(Delta_t, Tfinal + Delta_t, Delta_t)

        self.sol_n = Function(self.V)
        (u_n, _) = split(self.sol_n)
        (u, _) = split(self.sol)
        (v, _) = split(self.sol_test)
        F = (1/Delta_t) * inner(u - u_n, v) * dx - 0.5*(self.residual(self.sol, self.sol_test) + self.residual(self.sol_n, self.sol_test))
        F_prob = NonlinearVariationalProblem(F, self.sol, bcs=self.bcs)
        F_solver = NonlinearVariationalSolver(F_prob, solver_parameters=self.solver_params)

        # Solve state equation and compute stability
        self.solve(Constant(80))
        self.Re.assign(150)
        self.sol_n.assign(self.sol)
        self.sol.assign(self.sol_n)

        self.save_solution()
        Cd_medio = 0.
        Cl_medio = 0.
        print(List_t)
        for t in List_t:
            print("T = %.2e"%t)
            F_solver.solve()
            self.save_solution()
            self.sol_n.assign(self.sol)
            # if t % 1.0 == 0.0:
            Cd, Cl = self.compute_drag_lift()
            Cd_medio += Cd/Tfinal
            Cl_medio += Cl/Tfinal

        with stop_annotating():
            J_hat = ReducedFunctional(Cd_medio, Control(NS.f))
            for iteration in range(3):
                x, y = SpatialCoordinate(NS.mesh)
                x_c, y_c = -1.5, 0.0
                NS.g = exp(-100*(iteration + 1)*((x - x_c)**2 + (y - y_c)**2))
                NS.f = assemble(inner(as_vector([NS.g, NS.g]), v) * dx)
                J_hat(NS.f)
                J_hat.derivative(options={"riesz_representation": "l2"})


if __name__ == "__main__":
    NS = NS_simulation()
    # NS.solve(50)
    # Cd, _ = NS.compute_drag_lift()
    NS.transient()
