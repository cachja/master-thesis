# This script solves pressing of a rectangular piece of viscoelastic fluid with a free surface.
# It generates own mesh and loop over time schemes and time steps. For each combination, there
# is own instance of Fluid class It returns solution and pdf with plots of creep test and volume
# test for each combination of scheme and step.
# Ready to be run!

import os
import sys
import petsc4py

petsc4py.init(sys.argv)

from dolfin import *
import numpy as np
import mshr
import matplotlib.pyplot as plt

# Other FEniCS parameters
fileName = os.path.splitext(__file__)[0]
parameters['form_compiler']['optimize'] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4
parameters["ghost_mode"] = "shared_facet"
parameters["std_out_all_processes"] = False

# MPI setting
comm = MPI.comm_world
rank = MPI.rank(comm)


class Fluid(object):
    def __init__(self, name, mesh, bndry, t=0.0, dt=0.5, theta=0.5, *args, **kwargs):

        self.mesh = mesh
        self.bndry = bndry

        # Build function spaces (Taylor-Hood)
        VP2 = VectorElement("CG", mesh.ufl_cell(), 2)
        VP1 = VectorElement("CG", mesh.ufl_cell(), 1)
        P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
        EB = FiniteElement("CG", mesh.ufl_cell(), 1)  # Left Cauchy-Green tensor by components
        W = FunctionSpace(self.mesh, MixedElement([VP2, P1, EB, EB, EB, VP1]))
        self.W = W

        # Boundary conditions -------- let's do everything slip, except cylinder and free surface
        noslip = Constant((0, 0))
        slip = Constant(0)
        bc1 = DirichletBC(W.sub(0).sub(0), slip, bndry, 1)
        bc2 = DirichletBC(W.sub(0), noslip, bndry, 2)
        bc1_mesh = DirichletBC(W.sub(5).sub(0), slip, bndry, 1)
        bc2_mesh = DirichletBC(W.sub(5), noslip, bndry, 2)

        # Collect boundary conditions)
        self.bcs = [bc1, bc2, bc1_mesh, bc2_mesh]

        # Facet normal, identity tensor and boundary measure
        n = FacetNormal(mesh)
        I = Identity(mesh.geometry().dim())
        ds = Measure("ds", subdomain_data=bndry)

        # Define test function(s)
        (v_, p_, b11_, b12_, b22_, mesh_u_) = TestFunctions(W)

        # current unknown at time step t
        w = Function(W)
        (v, p, b11, b12, b22, mesh_u) = split(w)

        # previous known time step solution
        w0 = Function(W)
        (v0, p0, b110, b120, b220, mesh_u0) = split(w0)

        # Initial data
        w0ic = Expression(("0.0", "0.0", "0.0", "1.0", "0.0", "1.0", "0.0", "0.0"), degree=1)
        w0.assign(interpolate(w0ic, W))
        w.assign(interpolate(w0ic, self.W))

        # Assemble tensors
        B_ = as_tensor([[b11_, b12_], [b12_, b22_]])
        B = as_tensor([[b11, b12], [b12, b22]])
        B0 = as_tensor([[b110, b120], [b120, b220]])

        # Benchmark parameters
        nu = Constant(0.59)
        We = Constant(10.0)
        G = Constant(0.41 / We)
        rho = 1.0
        lambd = Constant(0.0)
        beta = Constant(0.0)
        a = Constant(1.0)  # gives Oldroyd upper convective derivative
        delta1 = Constant(1 / We)
        delta2 = Constant(0.0)

        # Define auxiliary variables for variational form
        def L(v): return grad(v)

        def F_hat(mesh_u): return I + grad(mesh_u)

        def J_hat(mesh_u): return det(F_hat(mesh_u))

        def F_hat_inv(mesh_u): return inv(F_hat(mesh_u))  # Maybe write an explicit formula later

        def Dv(v, mesh_u): return 0.5 * (L(v) * F_hat_inv(mesh_u) + F_hat_inv(mesh_u).T * L(v).T)

        def Wv(v, mesh_u): return 0.5 * (L(v) * F_hat_inv(mesh_u) - F_hat_inv(mesh_u).T * L(v).T)

        def T(p, v, mesh_u, B): return -p * I + 2.0 * nu * Dv(v, mesh_u) + G * ((1 - beta) * (B - I) + beta * (B - I) * B)

        # Define time derivatives
        ## time stepping parameters
        self.t = t
        self.theta = theta
        self.dt = dt
        self.k = Constant(1.0 / (dt * theta))

        def objective_derivative_NOtime(a, v, mesh_u, B):
            return dot(grad(B), F_hat_inv(mesh_u) * v) - a * (Dv(v, mesh_u) * B + B * Dv(v, mesh_u)) -\
                   (Wv(v, mesh_u) * B - B * Wv(v, mesh_u))

        def time_der(rho, mesh_u, mesh_u0, v, v0, v_):
            return rho * J_hat(mesh_u) * self.k * inner((v - v0),v_) * dx -\
                   rho * J_hat(mesh_u) * self.k * inner(dot(grad(v), F_hat_inv(mesh_u) * (mesh_u - mesh_u0)), v_) * dx

        # Define RHS
        def g(v, mesh_u0): return mesh_u0 + v / self.k

        force = Constant((0.0, 0.0))

        self.traction_magnitude = Constant(0.5)
        traction_tensor = self.traction_magnitude * as_tensor([[0.0, 0.0], [0.0, -1.0]])

        # Variational form without time derivative - plug current and previous time
        def EQ(v, p, B, mesh_u, v_, p_, B_, mesh_u_):
            Eq1 = J_hat(mesh_u) * tr(L(v) * F_hat_inv(mesh_u)) * p_ * dx
            Eq2 = J_hat(mesh_u) * rho * inner(L(v) * (F_hat_inv(mesh_u) * v), v_) * dx + \
                  J_hat(mesh_u) * inner(T(p, v, mesh_u, B) * F_hat_inv(mesh_u).T, grad(v_)) * dx - \
                  J_hat(mesh_u) * rho * inner(force, v_) * dx - \
                  J_hat(mesh_u) * inner(dot(traction_tensor * F_hat_inv(mesh_u), n), v_) * ds(5)
            Eq3 = J_hat(mesh_u) * inner(
                objective_derivative_NOtime(a, v, mesh_u, B) + delta1 * (B - I) + delta2 * (B - I) * B, B_) * dx + \
                  J_hat(mesh_u) * lambd * inner(grad(B), grad(B_)) * dx
            Eq4 = inner(L(mesh_u), L(mesh_u_)) * dx - \
                  inner(L(mesh_u) * n, mesh_u_) * (ds(4) + ds(5)) + \
                  inner(L(mesh_u_) * n, mesh_u - g(v, mesh_u0)) * (ds(4) + ds(5)) + \
                  Constant(1000) / h_min * inner(mesh_u - g(v, mesh_u0), mesh_u_) * (ds(4) + ds(5))
            return (Eq1 + Eq2 + Eq3 + Eq4)

        # combine variational forms with time derivative
        #
        #  dw/dt + F(t) = 0 is approximated as
        #  Implicit Glowinski three-step scheme (GL)
        #
        F = time_der(rho, mesh_u, mesh_u0, v, v0, v_) + \
            time_der(Constant(1.0), mesh_u, mesh_u0, B, B0, B_) + \
            EQ(v, p, B, mesh_u, v_, p_, B_, mesh_u_)
        DF = derivative(F, w)

        # Prepare solver
        self.w = w
        self.w0 = w0
        info("Solving problem of size: {0:d}".format(W.dim()))
        self.fluid_problem = NonlinearVariationalProblem(F, self.w, self.bcs, DF)
        self.fluid_solver = NonlinearVariationalSolver(self.fluid_problem)

        # Set solver parameters
        self.fluid_solver.parameters['newton_solver']['linear_solver'] = 'mumps'
        self.fluid_solver.parameters['newton_solver']['error_on_nonconvergence'] = False  # continue if diverged
        self.fluid_solver.parameters['newton_solver']['absolute_tolerance'] = 1e-8
        self.fluid_solver.parameters['newton_solver']['relative_tolerance'] = 1e-8
        self.fluid_solver.parameters['newton_solver']['maximum_iterations'] = 10
        self.fluid_solver.parameters['newton_solver']["krylov_solver"]['error_on_nonconvergence'] = False

        # Create files for storing solution
        self.vfile = XDMFFile(f"results_{name}/fluid_v.xdmf")
        self.mesh_ufile = XDMFFile(f"results_{name}/fluid_mesh_u.xdmf")
        self.vfile.parameters["flush_output"] = True
        self.mesh_ufile.parameters["flush_output"] = True

        # Prepare for the volume check
        self.VOL = J_hat(mesh_u) * dx
        self.Vref = assemble(self.VOL)

    def solve_step_GL(self):

        # solve the flow equations
        self.fluid_solver.solve()
        q = Constant((1.0 - self.theta) / self.theta)
        q0 = Constant((2 * self.theta - 1.0) / self.theta)
        self.w0.assign(q * self.w + q0 * self.w0)
        self.fluid_solver.solve()

        # Move to next time step
        self.t += self.dt
        if self.t > 5.0:
            self.traction_magnitude.assign(0.0)
        self.w0.assign(self.w)

    def solve_step_BE(self):

        # solve the flow equations
        self.fluid_solver.solve()

        # Move to next time step
        self.t += self.dt
        if self.t > 5.0:
            self.traction_magnitude.assign(0.0)
        self.w0.assign(self.w)

    def save(self):
        # Extract solutions:
        (v, p, b11, b12, b22, mesh_u) = self.w.split(True)
        v.rename("v", "velocity")
        mesh_u.rename("u", "deformation")

        # Save to file
        self.vfile.write(v, self.t)
        self.mesh_ufile.write(mesh_u, self.t)

        # Print most important piece of information
        self.max_u = MPI.max(mesh_f.mpi_comm(), np.max(mesh_u.vector().get_local()))
        info(f"Maximal mesh deformation: {self.max_u}")
        self.V = assemble(self.VOL)
        self.Vratio = self.V / self.Vref
        info(f"Volume: {self.V}, Referential Volume {self.Vref}, Ratio {self.Vratio}")


################### Main Part ########################

# Generate Mesh and prepare boundaries
Len = 2.0
Wid = 1.0

N = 60
mesh_f = RectangleMesh(Point(0, 0), Point(Len, Wid), int(Len * N), int(Wid * N), 'crossed')
mesh_f.init()

# Min/max cell diameters
h_min = MPI.min(mesh_f.mpi_comm(), mesh_f.hmin())
h_max = MPI.max(mesh_f.mpi_comm(), mesh_f.hmax())
info(f"h_min = {h_min} \t h_max = {h_max}")

# Define boundaries
Btraction = 5
Bfree = 4
BslipNO = 2
Bslipy = 1

bndry_f = MeshFunction("size_t", mesh_f, mesh_f.topology().dim() - 1, 0)
for f in facets(mesh_f):
    fp = f.midpoint()
    bndry_f[f] = 0
    if f.exterior():
        bndry_f[f] = 10
        if near(fp[0], 0.0): bndry_f[f] = Bslipy  # slip in y-direction
        if near(fp[0], Len): bndry_f[f] = Bslipy  # slip in y-direction
        if near(fp[1], 0.0): bndry_f[f] = BslipNO  # No slip
        if near(fp[1], Wid) and fp.distance(Point(Len / 2, Wid)) >= Len / 4: bndry_f[f] = Bfree  # top boundary
        if near(fp[1], Wid) and fp.distance(Point(Len / 2, Wid)) <= Len / 4: bndry_f[f] = Btraction

# Loop over methods and timesteps and save solutions, maximal deformation w.r.t. time and volume check w.r.t. time
method_list = ["BE", "GL"]
time_step_list = [1.0, 0.5, 0.1]

for method in method_list:
    for time_step in time_step_list:

        name = f"Scheme-{method}_dt-{time_step}"
        folder_name = f"results_{name}"
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        data_file = open(f"{folder_name}/data.txt", "w")

        t_end = 100.0
        if method == "GL": fluid = Fluid(name, mesh_f, bndry_f, t=0.0, dt=time_step, theta=0.29289321881345)
        if method == "BE": fluid = Fluid(name, mesh_f, bndry_f, t=0.0, dt=time_step, theta=1.0)

        graph_data = []
        volume_data = []
        t_data = []

        fluid.save()
        graph_data.append(fluid.max_u)
        volume_data.append(fluid.Vratio)
        t_data.append(fluid.t)

        while fluid.t <= t_end:

            info("t = {}".format(fluid.t))
            if method == "GL": fluid.solve_step_GL()
            if method == "BE": fluid.solve_step_BE()
            fluid.save()
            graph_data.append(fluid.max_u)
            volume_data.append(fluid.Vratio)
            t_data.append(fluid.t)

            data_file.write("t = " + str(float(fluid.t)) + "\t" + "volume_ratio = " + str(
                float(fluid.Vratio)) + "\t" + "max(|u|) = " + str(float(fluid.max_u)) + "\n")
            data_file.flush()
        data_file.close()

        # Plot graphs to pdf
        if rank == 0:
            plt.figure()

            plt.title('Creep test')
            plt.xlabel('t')
            plt.ylabel('max(|mesh_u|)')

            plt.plot(t_data, graph_data)

            plt.savefig(f"{folder_name}/creep_test.pdf", bbox_inches='tight')
            # plt.show()

        if rank == 0:
            plt.figure()

            plt.title('Volume test')
            plt.xlabel('t')
            plt.ylabel('V/V_ref')

            plt.plot(t_data, volume_data)

            plt.savefig(f"{folder_name}/volume_test.pdf", bbox_inches='tight')
            # plt.show()
