from firedrake import *
from firedrake.adjoint import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
plt.ion()  # Enable interactive mode for live plotting


class NS_simulation():
    def __init__(self):
        # file = "/Users/ddolci/tes_fire_install/firedrake_petsc_new/"
        # "test_new_adjoint_solver/"
        self.mesh = Mesh("symmetric_mesh.msh")
        # coordinates = np.loadtxt("solution/Re_200/mesh_final.txt",
        #                          delimiter=',')
        # self.mesh.coordinates.dat.data[:,:] = coordinates

        Vu = VectorFunctionSpace(self.mesh, "CG", 2)
        Vp = FunctionSpace(self.mesh, "CG", 1)
        self.V = MixedFunctionSpace([Vu, Vp])

        self.R = FunctionSpace(self.mesh, 'R', 0)
        self.Re = Function(self.R)
        self.sol = Function(self.V)
        self.sol_test = TestFunction(self.V)
        self.pvd = File("NS_200/sol.pvd")
        self.stream_bc = Function(self.R, val=1.0)
        self.wall_bc = Function(self.R, val=0.0)

        self.solver_params = {
                                "mat_type": "aij",
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
        self.bcs = [
            DirichletBC(self.V.sub(0),
                        (self.stream_bc, self.wall_bc),
                        (10, 12)),
            DirichletBC(self.V.sub(0), Constant((0., 0.)), (13))
        ]

    def solve(self, Re):
        self.Re.assign(Re)
        continue_annotation()
        F = self.residual(self.sol, self.sol_test)
        solve(F == 0, self.sol, bcs=self.bcs,
              solver_parameters=self.solver_params)

    def compute_drag_lift(self):
        (u, p) = split(self.sol)
        sigma = 1/self.Re*(grad(u) + grad(u).T) - p * Identity(2)
        drag = -2 * assemble(dot(sigma, FacetNormal(self.mesh))[0]*ds(13))
        lift = 2 * assemble(dot(sigma, FacetNormal(self.mesh))[1]*ds(13))
        return drag, lift

    def calculate_dominant_frequency(self, time_series, dt):
        # Number of sample points
        N = len(time_series)
        # Perform FFT
        yf = fft(time_series)
        xf = fftfreq(N, dt)[:N//2]
        # Compute the magnitude of the FFT
        magnitude = 2.0/N * np.abs(yf[0:N//2])
        # Find the index of the peak in the magnitude spectrum
        peak_index = np.argmax(magnitude)
        dominant_freq = xf[peak_index]
        return dominant_freq

    def save_solution(self):
        u = self.sol.sub(0)
        p = self.sol.sub(1)
        u.rename("u", "u")
        p.rename("p", "p")
        self.pvd.write(u, p)

    def transient(self):
        # Delta_t = 1/12
        Delta_t = 1/48
        Tfinal = 150.0

        self.sol_n = Function(self.V)
        (u_n, _) = split(self.sol_n)
        (u, _) = split(self.sol)
        (v, _) = split(self.sol_test)
        F = (1/Delta_t) * inner(u - u_n, v) * dx - 0.5*(self.residual(self.sol, self.sol_test) + self.residual(self.sol_n, self.sol_test))
        F_prob = NonlinearVariationalProblem(F, self.sol, bcs=self.bcs)
        F_solver = NonlinearVariationalSolver(F_prob, solver_parameters=self.solver_params)

        # Solve state equation and compute stability
        self.solve(Constant(50))
        self.Re.assign(80. + 0.01)
        self.sol_n.assign(self.sol)
        self.sol.assign(self.sol_n)

        self.save_solution()
        Cd_medio = 0.
        Cl_medio = 0.
        
        # Lists to store time series for periodicity analysis
        time_list = []
        Cd_list = []
        Cl_list = []

        for i, t in enumerate(np.arange(Delta_t, Tfinal + Delta_t, Delta_t)):
            print("T = %.2e (step %d/%d)" % (t, i+1, int(Tfinal/Delta_t)))
            F_solver.solve()
            if i % 1000 == 0:
                print("Saving solution at T = %.2e" % t)
                self.save_solution()
                # Save cd and cl curves
                plt.plot(time_list, Cd_list, label='Cd')
                plt.plot(time_list, Cl_list, label='Cl')
                plt.xlabel('Time')
                plt.ylabel('Coefficient')
                plt.legend()
                plt.savefig(f'NS_200/Cd_Cl_up_to_{t:.2f}s.png')
                plt.clf()

            self.sol_n.assign(self.sol)

            Cd, Cl = self.compute_drag_lift()
            Cd_medio += Cd/Tfinal
            Cl_medio += Cl/Tfinal

            # Store for periodicity analysis
            time_list.append(float(t))
            Cd_list.append(float(Cd))
            Cl_list.append(float(Cl))
        
        # Calculate dominant frequency from Cl time series
        dominant_freq = self.calculate_dominant_frequency(Cl_list, Delta_t)
        # Calculate period from dominant frequency
        period_fft = 1.0 / dominant_freq
        print(f"Dominant frequency: {dominant_freq:.6f}, Period from FFT: {period_fft:.6f}")

        # Run the solver for one period to capture periodic cycle
        try:
            period_times = np.arange(0, period_fft, Delta_t)
        except ValueError:
            period_times = np.arange(0, 1.0, Delta_t)  # Fallback to 1s if period_fft is invalid
        
        Cd_period_list = []
        Cl_period_list = []
        continue_annotation()
        cd_mean = 0.0
        cl_mean = 0.0
        for j, t_period in enumerate(period_times):
            print(f"Periodic cycle: T = %.2e (step %d/%d)" % (t_period, j+1, len(period_times)))
            F_solver.solve()
            self.sol_n.assign(self.sol)
            Cd_p, Cl_p = self.compute_drag_lift()
            cd_mean += Cd_p/len(period_times)
            cl_mean += Cl_p/len(period_times)
            Cd_period_list.append(float(Cd_p))
            Cl_period_list.append(float(Cl_p))
        # Save figures during periodic cycle
        plt.plot(period_times, Cd_period_list, label='Cd')
        plt.plot(period_times, Cl_period_list, label='Cl')
        plt.xlabel('Time')
        plt.ylabel('Coefficient')
        plt.legend()
        plt.title('Drag and Lift Coefficients Over One Period')
        plt.grid(True)
        plt.savefig('NS_200/drag_lift_coefficients_periodic_cycle.png')

        # print Cd and Cl mean values in a period
        print(f"Mean Drag Coefficient Cd over one period: {cd_mean:.6f}")
        print(f"Mean Lift Coefficient Cl over one period: {cl_mean:.6f}")
        J_hat = ReducedFunctional(cl_mean, Control(self.Re))
        get_working_tape().progress_bar = ProgressBar
        taylor_test(J_hat, self.Re, Function(self.R, val=0.01), dJdm=0.)
        quit()
        with stop_annotating():
            derivative = J_hat.derivative()

            # Extract derivative value
            try:
                deriv_value = float(derivative.dat.data[0])
            except AttributeError:
                deriv_value = float(derivative)

            print(f"  dJ/dRe = {deriv_value:.8e}")


if __name__ == "__main__":
    NS = NS_simulation()
    # NS.solve(50)
    # Cd, _ = NS.compute_drag_lift()
    NS.transient()

# Re80
# Mean Drag Coefficient Cd over one period: 1.598600
# Mean Lift Coefficient Cl over one period: -0.006011
#   dJ/dRe = -2.56001325e-03
# Re=80.01
# Mean Drag Coefficient Cd over one period: 1.598571
# Mean Lift Coefficient Cl over one period: -0.006036