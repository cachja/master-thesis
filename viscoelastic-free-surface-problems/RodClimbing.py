# This script solves Rod Climbing problem for the classical Oldroyd-B model,
# but any convex combination of Oldroyd-B with Giesekus model with stress
# diffusion is possible (see # Benchmark parameters).
# The script generates own mesh and initialize an instance of a class that
# defines properties of the fluid, solver and more FEniCS needs.
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
parameters["std_out_all_processes"] = False

# MPI setting
comm = MPI.comm_world
rank = MPI.rank(comm)


class Fluid(object):
    def __init__(self, name, mesh, bndry, t=0.0, dt=0.5, theta=0.5, *args, **kwargs):
        self.mesh = mesh
        self.bndry = bndry

        # Build function spaces (Taylor-Hood)
        P2 = FiniteElement("CG", mesh.ufl_cell(), 2)
        P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
        EB = FiniteElement("CG", mesh.ufl_cell(), 1)  # Left Cauchy-Green tensor by components
        W = FunctionSpace(self.mesh, MixedElement([P2, P2, P2, P1, EB, EB, EB, EB, EB, EB, P2, P2]))
        self.W = W

        # Boundary conditions -------- everything no-slip, except container wall in z, cylinder and free surface
        noslip = Constant(0)
        # Container wall
        bc1_r = DirichletBC(W.sub(0), noslip, bndry, 1)
        bc1_phi = DirichletBC(W.sub(1), noslip, bndry, 1)
        # Container bottom
        bc2_r = DirichletBC(W.sub(0), noslip, bndry, 2)
        bc2_phi = DirichletBC(W.sub(1), noslip, bndry, 2)
        bc2_z = DirichletBC(W.sub(2), noslip, bndry, 2)
        # Symmetry wall
        bc3_r = DirichletBC(W.sub(0), noslip, bndry, 3)
        bc3_phi = DirichletBC(W.sub(1), noslip, bndry, 3)
        # Cylinder mantle
        self.vc_phi = Expression(("-VelRef*x[0]/R"), VelRef=VelRef, R=radius, degree=8)
        bc5_r = DirichletBC(W.sub(0), noslip, bndry, 5)
        bc5_phi = DirichletBC(W.sub(1), self.vc_phi, bndry, 5)
        # Cylinder bottom
        bc6_r = DirichletBC(W.sub(0), noslip, bndry, 6)
        bc6_phi = DirichletBC(W.sub(1), self.vc_phi, bndry, 6)
        bc6_z = DirichletBC(W.sub(2), noslip, bndry, 6)
        # Corresponding mesh BC
        bc1_mesh_r = DirichletBC(W.sub(10), noslip, bndry, 1)
        bc2_mesh_r = DirichletBC(W.sub(10), noslip, bndry, 2)
        bc2_mesh_z = DirichletBC(W.sub(11), noslip, bndry, 2)
        bc3_mesh_r = DirichletBC(W.sub(10), noslip, bndry, 3)
        bc5_mesh_r = DirichletBC(W.sub(10), noslip, bndry, 5)
        bc6_mesh_r = DirichletBC(W.sub(10), noslip, bndry, 6)
        bc6_mesh_z = DirichletBC(W.sub(11), noslip, bndry, 6)

        self.bcs = [bc1_r, bc1_phi, bc1_mesh_r, bc2_r, bc2_phi, bc2_z, bc2_mesh_r, bc2_mesh_z, bc3_r, bc3_phi, bc3_mesh_r, bc5_phi, bc6_phi, bc5_r, bc5_mesh_r, bc6_r, bc6_z, bc6_mesh_r, bc6_mesh_z]

        # Facet normal, identity tensor and boundary measure
        n_2D = FacetNormal(mesh)
        self.n_2D = n_2D
        n = as_vector([n_2D[0], 0, n_2D[1]])
        I = Identity(mesh.geometry().dim() + 1)
        ds = Measure("ds", subdomain_data=bndry)

        # Define test function(s)
        (v1_, v2_, v3_, p_, b11_, b12_, b22_, b13_, b23_, b33_, mesh_u1_, mesh_u3_) = TestFunctions(W)

        # current unknown at time step t
        w = Function(W)
        (v1, v2, v3, p, b11, b12, b22, b13, b23, b33, mesh_u1, mesh_u3) = split(w)

        # previous known time step solution
        w0 = Function(W)
        (v10, v20, v30, p0, b110, b120, b220, b130, b230, b330, mesh_u10, mesh_u30) = split(w0)

        # Initial data
        w0ic = Expression(("0.0", "0.0", "0.0", "0.0", "1.0", "0.0", "1.0", "0.0", "0.0", "1.0", "0.0", "0.0"), degree=1)
        w0.assign(interpolate(w0ic, W))
        w.assign(interpolate(w0ic, self.W))

        # Assemble vectors
        v = as_vector([v1, v2, v3])
        v_ = as_vector([v1_, v2_, v3_])
        v0 = as_vector([v10, v20, v30])

        mesh_u = as_vector([mesh_u1, 0.0, mesh_u3])
        mesh_u_ = as_vector([mesh_u1_, 0.0, mesh_u3_])
        mesh_u0 = as_vector([mesh_u10, 0.0, mesh_u30])

        # Assemble tensors
        B_ = as_tensor([[b11_, b12_, b13_], [b12_, b22_, b23_], [b13_, b23_, b33_]])
        B = as_tensor([[b11, b12, b13], [b12, b22, b23], [b13, b23, b33]])
        B0 = as_tensor([[b110, b120, b130], [b120, b220, b230], [b130, b230, b330]])

        # Benchmark parameters
        nu = Constant(30.0)
        G = Constant(2.0)
        We = Constant(1.0)
        rho = Constant(1000.0)
        lambd = Constant(0.0)
        beta = Constant(0.0)
        a = Constant(1.0)  # gives Oldroyd upper convected derivative
        delta1 = Constant(1.0 / We)
        delta2 = Constant(0.0)

        info(f"Reynolds number = {float(rho) / float(nu) * radius * VelRef}")

        # Define cylindrical gradient for scalar, vector a 2nd order tensorial field
        x = Expression(("x[0]"), degree=2)  # x[0] = r, x[1] = z

        def L_scal(p):  # p = p(r,z)
            return as_vector([p.dx(0), 0.0, p.dx(1)])

        def L_vec(v):
            cylindrical_second_column = as_vector([-v[1] / x, v[0] / x, 0.0])
            cylindrical_nabla_vector = as_tensor([v.dx(0), cylindrical_second_column, v.dx(1)]).T
            return cylindrical_nabla_vector

        def L_ten(B):
            cylindrical_second_matrix = -as_tensor([[2 * B[0][1], B[1][1] - B[0][0], B[1][2]], [B[1][1] - B[0][0], -2 * B[0][1], -B[0][2]],[B[1][2], -B[0][2], 0.0]]) / x
            cylindrical_nabla_tensor = as_tensor([B.dx(0), cylindrical_second_matrix, B.dx(1)])
            return cylindrical_nabla_tensor

        def L_tenOnVec(B, v):
            first_matrix = as_tensor([[dot(L_scal(B[0][0]), v), dot(L_scal(B[0][1]), v), dot(L_scal(B[0][2]), v)],
                                      [dot(L_scal(B[0][1]), v), dot(L_scal(B[1][1]), v), dot(L_scal(B[1][2]), v)],
                                      [dot(L_scal(B[0][2]), v), dot(L_scal(B[1][2]), v), dot(L_scal(B[2][2]), v)]])
            cylindrical_second_matrix = -as_tensor([[2 * B[0][1], B[1][1] - B[0][0], B[1][2]], [B[1][1] - B[0][0], -2 * B[0][1], -B[0][2]],[B[1][2], -B[0][2], 0.0]]) / x * v[1]
            return first_matrix + cylindrical_second_matrix

        # Define auxiliary variables for variational form
        def F_hat(mesh_u): return I + L_vec(mesh_u)

        def J_hat(mesh_u): return det(F_hat(mesh_u))

        def F_hat_inv(mesh_u): return inv(F_hat(mesh_u))

        def Dv(v, mesh_u): return 0.5 * (L_vec(v) * F_hat_inv(mesh_u) + F_hat_inv(mesh_u).T * L_vec(v).T)

        def Wv(v, mesh_u): return 0.5 * (L_vec(v) * F_hat_inv(mesh_u) - F_hat_inv(mesh_u).T * L_vec(v).T)

        # Define objective time derivatives
        ## time stepping parameters
        self.t = t
        self.theta = theta
        self.dt = dt
        self.k = Constant(1.0 / (dt * theta))

        def objective_derivative_NOtime(a, v, mesh_u, B):
            return L_tenOnVec(B, F_hat_inv(mesh_u) * v) - a * (Dv(v, mesh_u) * B + B * Dv(v, mesh_u)) -\
                   (Wv(v, mesh_u) * B - B * Wv(v, mesh_u))

        def time_der_vec(rho, mesh_u, mesh_u0, v, v0, v_):
            return x * rho * J_hat(mesh_u) * self.k * inner((v - v0), v_) * dx -\
                   x * rho * J_hat(mesh_u) * self.k * inner(dot(L_vec(v), F_hat_inv(mesh_u) * (mesh_u - mesh_u0)), v_) * dx

        def time_der_ten(rho, mesh_u, mesh_u0, B, B0, B_):
            return x * rho * J_hat(mesh_u) * self.k * inner((B - B0), B_) * dx -\
                   x * rho * J_hat(mesh_u) * self.k * inner(L_tenOnVec(B, F_hat_inv(mesh_u) * (mesh_u - mesh_u0)), B_) * dx

        # Define RHS
        force = Constant((0.0, 0.0, -10.0))

        def g(v, mesh_u0): return mesh_u0 + v / self.k

        # Define Cauchy stress
        def T(p, v, mesh_u, B): return -p * I + 2.0 * nu * Dv(v, mesh_u) + a * G * ((1 - beta) * (B - I) + beta * (B - I) * B)

        # Variational form without time derivatives
        def EQ(v, p, B, mesh_u, v_, p_, B_, mesh_u_):
            Eq1 = x * J_hat(mesh_u) * tr(L_vec(v) * F_hat_inv(mesh_u)) * p_ * dx
            Eq2 = x * J_hat(mesh_u) * rho * inner(L_vec(v) * (F_hat_inv(mesh_u) * v), v_) * dx +\
                  x * J_hat(mesh_u) * inner(dot(T(p, v, mesh_u, B), F_hat_inv(mesh_u).T), L_vec(v_)) * dx -\
                  x * J_hat(mesh_u) * rho * inner(force, v_) * dx
            Eq3 = x * J_hat(mesh_u) * inner(objective_derivative_NOtime(a, v, mesh_u, B) + delta1 * (B - I) + delta2 * (B - I) * B, B_) * dx +\
                  x * J_hat(mesh_u) * lambd * inner(L_ten(B), L_ten(B_)) * dx
            Eq4 = x * inner(L_vec(mesh_u), L_vec(mesh_u_)) * dx -\
                  x * inner(L_vec(mesh_u) * n, mesh_u_) * ds(4) +\
                  x * inner(L_vec(mesh_u_) * n, mesh_u -g(v, mesh_u0)) * ds(4) +\
                  x * Constant(1000.0) / h_min * inner(mesh_u - g(v, mesh_u0), mesh_u_) * ds(4)
            return Eq1 + Eq2 + Eq3 + Eq4

        # combine variational forms with time derivative
        #
        #  dw/dt + F(t) = 0 is approximated as
        #  Implicit Glowinski three-step scheme (GL)
        #  Or Backward Euler scheme (BE)
        #
        F = time_der_vec(rho, mesh_u, mesh_u0, v, v0, v_) +\
            time_der_ten(Constant(1.0), mesh_u, mesh_u0, B, B0, B_) +\
            EQ(v, p, B, mesh_u, v_, p_, B_, mesh_u_)
        DF = derivative(F, w)
        self.F = F

        # Prepare solver
        self.w = w
        self.w0 = w0
        info("Solving problem of size: {0:d}".format(W.dim()))
        self.fluid_problem = NonlinearVariationalProblem(F, self.w, self.bcs, DF)
        self.fluid_solver = NonlinearVariationalSolver(self.fluid_problem)

        # Set solver parameters
        self.fluid_solver.parameters['newton_solver']['linear_solver'] = 'mumps'
        self.fluid_solver.parameters['newton_solver']['error_on_nonconvergence'] = True
        self.fluid_solver.parameters['newton_solver']['absolute_tolerance'] = 5e-9
        self.fluid_solver.parameters['newton_solver']['relative_tolerance'] = 5e-9
        self.fluid_solver.parameters['newton_solver']['maximum_iterations'] = 10
        self.fluid_solver.parameters['newton_solver']["krylov_solver"]['error_on_nonconvergence'] = True

        # Prepare for the volume check
        self.VOL = 2 * 3.1415 * x * J_hat(as_vector([mesh_u[0], 0.0, mesh_u[2]])) * dx
        self.Vref = assemble(self.VOL)

        # Create files for storing solution
        self.vfile = XDMFFile(f"results_{name}/fluid_v.xdmf")
        self.mesh_ufile = XDMFFile(f"results_{name}/fluid_mesh_u.xdmf")
        self.v_phifile = XDMFFile(f"results_{name}/fluid_v_phi.xdmf")
        self.mesh_u_phifile = XDMFFile(f"results_{name}/fluid_mesh_u_phi.xdmf")
        self.pfile = XDMFFile(f"results_{name}/fluid_p.xdmf")
        self.vfile.parameters["flush_output"] = True
        self.mesh_ufile.parameters["flush_output"] = True
        self.v_phifile.parameters["flush_output"] = True
        self.mesh_u_phifile.parameters["flush_output"] = True
        self.pfile.parameters["flush_output"] = True
        self.rfile = XDMFFile(f"results_{name}/fluid_res.xdmf")
        self.rfile.parameters["flush_output"] = True
        self.data_file = open(f"results_{name}/data.txt", "w")
        self.volume_data = []
        self.t_data = []

    def solve_step_GL(self):  # change theta = 0.29289321881345 in fluid=Fluid
        # solve the flow equations
        ## solve first step
        self.fluid_solver.solve()
        ## actualize solution
        q = Constant((1.0 - self.theta) / self.theta)
        q0 = Constant((2 * self.theta - 1.0) / self.theta)
        self.w0.assign(q * self.w + q0 * self.w0)
        self.w.assign(q*self.w+q0*self.w0)
        ## solve second step
        self.fluid_solver.solve()

        # Move to next time step
        self.t += self.dt
        self.w0.assign(self.w)

    def solve_step_BE(self):  # change theta = 1.0 in fluid=Fluid
        # solve the flow equations
        self.fluid_solver.solve()

        # Move to next time step
        self.t += self.dt
        self.w0.assign(self.w)

    def save(self):
        # Extract solutions:
        (v1, v2, v3, p, b11, b12, b22, b13, b23, b33, mesh_u1, mesh_u3) = self.w.split(True)
        # Make vectors that are usefull for Paraview
        v_2D = as_vector([v1, v3])
        mesh_u_2D = as_vector([mesh_u1, mesh_u3])

        # Save to file
        v_2D_projected = project(v_2D, FunctionSpace(mesh_f, VectorElement("CG", mesh_f.ufl_cell(), 2)))
        mesh_u_2D_projected = project(mesh_u_2D, FunctionSpace(mesh_f, VectorElement("CG", mesh_f.ufl_cell(), 2)))
        v_2D_projected.rename("v", "velocity")
        mesh_u_2D_projected.rename("u", "deformation")
        v2.rename("v_phi", "velocity")
        p.rename("p", "deformation")
        self.vfile.write(v_2D_projected, self.t)
        self.mesh_ufile.write(mesh_u_2D_projected, self.t)
        self.v_phifile.write(v2, self.t)
        self.pfile.write(p, self.t)

        # Print most important piece of information
        self.max_u = MPI.max(mesh_f.mpi_comm(), np.max(mesh_u3.vector().get_local()))
        info(f"Maximal mesh deformation in z: {self.max_u}")
        self.V = assemble(self.VOL)
        info(f"Volume: {self.V}, Referential Volume {self.Vref}, ratio {self.V / self.Vref}")

        # Check the Courant number
        v_max = MPI.max(mesh_f.mpi_comm(), np.abs(v_2D_projected.vector().get_local()).max())
        Courant = v_max * self.dt / h_min
        info(f"Courant number: {Courant}")

        # Store volume data
        self.volume_data.append(self.V / self.Vref)
        self.t_data.append(self.t)

        self.data_file.write(
            "t = " + str(float(fluid.t)) + "\t" + "volume_ratio = " + str(float(self.V / self.Vref)) + "\n")
        self.data_file.flush()

    # One can store residuum for each variable and save to the file
    def get_residuum(self):
        r = Function(self.W)
        assemble(self.F, tensor=r.vector())
        [bc.apply(r.vector()) for bc in self.bcs]
        r.rename("r", "residuum")
        self.rfile.write(r, self.t)


###### Main Part ######

# Set dimensions of the rod climbing problem
lenght = 0.1
width = 0.1
height = 0.1 / 2.0
radius = 0.01
cylinder_bottom = 0.02
RPM = 100.0
VelRef = 2.0 * 3.1415 * radius * RPM / 60.0

# Generate Mesh and prepare boundaries
N = 50  # Leads to the 200,000 DoFs with actual choice of refinement
fluid_domain = mshr.Rectangle(Point(0.0, 0.0), Point(lenght / 2, height))
cylinder_temp = mshr.Rectangle(Point(0.0, cylinder_bottom), Point(radius, height))
domain = (fluid_domain - cylinder_temp)
mesh_f = mshr.generate_mesh(domain, N)
mesh_f.init()

# Local refinement of the mesh
local_refinement_iterations = 4
for k in range(local_refinement_iterations):  # Refine free surface
    info("refinement level {}".format(k))
    cf = MeshFunction('bool', mesh_f, mesh_f.topology().dim(), False)
    for c in cells(mesh_f):
        for vert in vertices(c):
            if vert.point()[1] >= height - height / 160.0: cf[c] = True
    mesh_f = refine(mesh_f, cf, redistribute=True)
for k in range(0):  # Refine Cylinder if needed
    info("refinement level {}".format(k))
    cf = MeshFunction('bool', mesh_f, mesh_f.topology().dim(), False)
    for c in cells(mesh_f):
        for vert in vertices(c):
            if vert.point()[0] <= 1.05 * radius and vert.point()[1] >= 0.95 * cylinder_bottom: cf[c] = True
    mesh_f = refine(mesh_f, cf, redistribute=True)
mesh_f.init()

# Min/max cell diameters
h_min = MPI.min(mesh_f.mpi_comm(), mesh_f.hmin())
h_max = MPI.max(mesh_f.mpi_comm(), mesh_f.hmax())
info(f"h_min = {h_min} \t h_max = {h_max}")

# Plot mesh
plot(mesh_f)
plt.show()

info("Mesh's ready and waiting")

# Define boundaries
bndry_f = MeshFunction("size_t", mesh_f, mesh_f.topology().dim() - 1, 0)
for f in facets(mesh_f):
    mp = f.midpoint()
    bndry_f[f] = 0
    if f.exterior():
        if near(mp[0], lenght / 2):  # wall
            bndry_f[f] = 1
        elif near(mp[1], 0.0):  # bottom
            bndry_f[f] = 2
        elif near(mp[0], 0.0):  # symmetry wall
            bndry_f[f] = 3
        elif near(mp[1], height):  # FREE SURFACE
            bndry_f[f] = 4
        elif near(mp[1], cylinder_bottom) and mp[0] <= radius + 0.000001:  # cylinder bottom
            bndry_f[f] = 6
        elif (mp[0] < radius + 1e-5) and (mp[1] >= cylinder_bottom - 0.000001):  # cylinder wall
            bndry_f[f] = 5

# Set name for the file with output
name = "Rod-climbing"

# Save boundaries to the file
with XDMFFile("results_%s/mesh_bndry.xdmf" % name) as f:
    f.write(bndry_f)

# Define end time of the simulation
t_end = 6.04

# Initializing an instance of a class that defines properties of the fluid, solver, etc.
fluid = Fluid(name, mesh_f, bndry_f, t=0.0, dt=0.01, theta=0.29289321881345)

time_step = 1
while fluid.t <= t_end:
    info("t = {}".format(fluid.t))
    # Choose time stepping scheme - Implicit Glowinski three-step (GL) scheme reccomended
    # Backward Euler (BE) is cheaper but does not preserve volume as well as (GL)
    fluid.solve_step_GL() # set fluid=Fluid(...theta = 0.29289321881345)
    # fluid.solve_step_BE() # set fluid=Fluid(...theta = 1.0)
    fluid.save()
    fluid.get_residuum()
    time_step += 1

# At the end of the simulation check volume preservation throughout computation
if rank == 0:
    plt.figure()

    plt.title('Volume test')
    plt.xlabel('t')
    plt.ylabel('V/V_ref')

    plt.plot(fluid.t_data, fluid.volume_data)

    plt.savefig("results_%s/volume_test.pdf" % name, bbox_inches='tight')
    plt.show()
