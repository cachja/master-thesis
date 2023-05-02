# This script solves the axisymmetric shearflow problem for a viscoelastic fluid
# described by the classical Giesekus model. It generates own mesh, solves problem
# and saves solution to a file for We in range (0.01, 1.0).
# Ready to be run!

from dolfin import *
import mshr
import numpy as np

# PETSc parameters
PETScOptions.set('mat_mumps_icntl_14', 1000)  # work array, multiple of estimate to allocate
PETScOptions.set('mat_mumps_icntl_24', 1)  # detect null pivots
PETScOptions.set('mat_mumps_cntl_1', 1.0)  # pivoting threshold, this solves to machine precision

# MPI setting
comm = MPI.comm_world
rank = MPI.rank(comm)

# Other FEniCS parameters
set_log_level(LogLevel.INFO if rank == 0 else LogLevel.INFO)
parameters["std_out_all_processes"] = False
parameters['form_compiler']['optimize'] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4
parameters["refinement_algorithm"] = "plaza_with_parent_facets"
parameters["ghost_mode"] = "shared_facet"

# Save solution in VTK format
pfile = XDMFFile(comm, "results_stationary_VE/p.xdmf")  # pressure file
v1file = XDMFFile(comm, "results_stationary_VE/v_rz.xdmf")  # (v_r,v_z) file
v2file = XDMFFile(comm, "results_stationary_VE/v_phi.xdmf")  # v_phi file

# Create mesh
## Proportions
height = 0.9
h_inner = 6 / 20 * height
r_inner = 0.25
r_outer = 0.75
center_outer_down = Point(0, -height / 2)
corner_outer_up = Point(r_outer, height / 2)
center_inner_down = Point(0, -h_inner / 2)
corner_inner_up = Point(r_inner, h_inner / 2)

# Generate mesh
mesh_refinement = 90  # Leads to the 200,000 DoFs
fluid_domain = mshr.Rectangle(center_outer_down, corner_outer_up)
cylinder = mshr.Rectangle(center_inner_down, corner_inner_up)
domain = fluid_domain - cylinder
mesh = mshr.generate_mesh(domain, mesh_refinement)
mesh.init()

# Min/max cell diameters
h_min = MPI.min(mesh.mpi_comm(), mesh.hmin())
h_max = MPI.max(mesh.mpi_comm(), mesh.hmax())
print(f"h_min = {h_min} \t h_max = {h_max}")

# Prepare boundaries
bndry = MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)

# Define finite elements
Ev = FiniteElement("CG", mesh.ufl_cell(), 2)  # Velocity
EB = FiniteElement("CG", mesh.ufl_cell(), 1)  # Left Cauchy-Green tensor
Ep = FiniteElement("CG", mesh.ufl_cell(), 1)  # Pressure

# Coupling I - Build function spaces (Taylor-Hood) for coupled problem
W = FunctionSpace(mesh, MixedElement([Ev, Ev, Ev, Ep, EB, EB, EB, EB, EB, EB]))

# Define boundaries
for f in facets(mesh):
    mp = f.midpoint()
    bndry[f] = 100
    if f.exterior():
        if near(mp[1], -height / 2):  # outer lower wall
            bndry[f] = 1
        elif near(mp[1], height / 2):  # outer upper wall
            bndry[f] = 1
        elif near(mp[0], r_outer):  # outer mantle
            bndry[f] = 1
        elif near(mp[1], -h_inner / 2) and mp.distance(center_inner_down) <= r_inner + 1e-2:  # inner lower wall
            bndry[f] = 4
        elif near(mp[1], h_inner / 2) and mp.distance(Point(0.0, h_inner / 2)) <= r_inner + 1e-2:  # inner upper wall
            bndry[f] = 4
        elif mp.distance(Point(0, mp[1])) <= r_inner + 1e-2 and abs(mp[1]) <= h_inner / 2:  # inner mantle
            bndry[f] = 4
        elif near(mp[0], 0):  # symmetry wall
            bndry[f] = 7

# Save boundaries to the file
with XDMFFile("results_stationary_VE/mesh_bndry.xdmf") as f:
    f.write(bndry)

## Facet normal and boundary measure
n = FacetNormal(mesh)
ds = Measure("ds", subdomain_data=bndry)
h = CellDiameter(mesh)

# Define boundary conditions
omega = 0.2
noslip = Constant(0)
bc1 = DirichletBC(W.sub(0), noslip, bndry, 1)
bc2 = DirichletBC(W.sub(1), noslip, bndry, 1)
bc3 = DirichletBC(W.sub(2), noslip, bndry, 1)
bc_phi = Expression(("x[0]*omega"), omega=omega, degree=8)
bc4 = DirichletBC(W.sub(0), noslip, bndry, 4)
bc5 = DirichletBC(W.sub(1), bc_phi, bndry, 4)
bc6 = DirichletBC(W.sub(2), noslip, bndry, 4)

# Coupling II - we combine boundary conditions for velocity and cauchy tensor
bcs = [bc1, bc2, bc3, bc4, bc5, bc6]

# Coupling III - W is a function of velocity, left Cauchy-Green tensor and pressure from W space
(v1_, v2_, v3_, p_, b11_, b12_, b22_, b13_, b23_, b33_) = TestFunctions(W)
w = Function(W)
(v1, v2, v3, p, b11, b12, b22, b13, b23, b33) = split(w)

# Assemble vectors
v = as_vector([v1, v2, v3])
v_ = as_vector([v1_, v2_, v3_])

# Assemble matrices
B_ = as_tensor([[b11_, b12_, b13_], [b12_, b22_, b23_], [b13_, b23_, b33_]])
B = as_tensor([[b11, b12, b13], [b12, b22, b23], [b13, b23, b33]])

# Initial data
w0ic = Expression(("0.0", "0.0", "0.0", "0.0", "1.0", "0.0", "1.0", "0.0", "0.0", "1.0"), degree=1)
w.assign(interpolate(w0ic, W))

###### Giesekus variant model ######

# Prepare cylindrical coordinates
x = Expression(("x[0]"), degree=2)  # x[0] = r, x[1] = z


# Redefine gradients
def L_scal(p):  # p = p(r,z)
    return as_vector([p.dx(0), 0.0, p.dx(1)])


def L_vec(v):
    cylindrical_second_column = as_vector([-v[1] / x, v[0] / x, 0])
    cylindrical_nabla_vector = as_tensor([v.dx(0), cylindrical_second_column, v.dx(1)]).T
    return cylindrical_nabla_vector


def L_ten(B):
    # cylindrical_second_matrix = -as_tensor([[2*b12,b22-b11,b23],[b22-b11,2*b12,-b13],[b23,-b13,0]])/x
    cylindrical_second_matrix = -as_matrix(
        [[2 * B[0][1], B[1][1] - B[0][0], B[1][2]], [B[1][1] - B[0][0], -2 * B[0][1], -B[0][2]],[B[1][2], -B[0][2], 0.0]]) / x
    cylindrical_nabla_tensor = as_tensor([B.dx(0), cylindrical_second_matrix, B.dx(1)])
    return cylindrical_nabla_tensor


def L_tenOnV(B, v):
    first_matrix = as_matrix([[dot(L_scal(B[0][0]), v), dot(L_scal(B[0][1]), v), dot(L_scal(B[0][2]), v)],
                              [dot(L_scal(B[0][1]), v), dot(L_scal(B[1][1]), v), dot(L_scal(B[1][2]), v)],
                              [dot(L_scal(B[0][2]), v), dot(L_scal(B[1][2]), v), dot(L_scal(B[2][2]), v)]])
    cylindrical_second_matrix = -as_matrix([[2 * B[0][1], B[1][1] - B[0][0], B[1][2]], [B[1][1] - B[0][0], -2 * B[0][1], -B[0][2]],[B[1][2], -B[0][2], 0.0]]) / x
    return first_matrix + cylindrical_second_matrix * v[1]


def Dv(v): return 0.5 * (L_vec(v) + L_vec(v).T)


def Wv(v): return 0.5 * (L_vec(v) - L_vec(v).T)


# Benchmark parameters
We = Constant(0.1)
G = Constant(0.1)
lambd = Constant(0.0)
a = Constant(1.0)  # gives Oldroyd upper convective derivative
delta = Constant(0.5 / We)
Re = Constant(5.0)

# Define variational problem
I = Identity(mesh.geometry().dim() + 1)
T = -p / Re * I + 2.0 / Re * Dv(v) + G * (B - 0.5 * tr(B) * I)


def objective_derivative_NOtime(a, v, B):
    return L_tenOnV(B, v) - a * (Dv(v) * B + B * Dv(v)) - (Wv(v) * B - B * Wv(v))


Eq1 = tr(L_vec(v)) * p_ * dx
Eq2 = inner(L_vec(v) * v, v_) * dx + inner(T, L_vec(v_)) * dx
Eq3 = inner(objective_derivative_NOtime(a, v, B), B_) * dx + delta * inner(B * B - I, B_) * dx

# Coupling VI - we sum continuity and momentum eqs. and solve viscoelastic problem together
Eq = Eq1 + Eq2 + Eq3

## prepare solver 
info("Solving problem of size: {0:d}".format(W.dim()))
problem = NonlinearVariationalProblem(Eq, w, bcs, derivative(Eq, w))
solver = NonlinearVariationalSolver(problem)

## set solvers parameters
solver.parameters['newton_solver']['linear_solver'] = 'mumps'
solver.parameters['newton_solver']['error_on_nonconvergence'] = False  # continue if diverged
solver.parameters['newton_solver']['absolute_tolerance'] = 1e-10
solver.parameters['newton_solver']['relative_tolerance'] = 1e-10
solver.parameters['newton_solver']['maximum_iterations'] = 10
solver.parameters['newton_solver']["krylov_solver"]['error_on_nonconvergence'] = False

###### Main part ######
# Prescribe Weissenberg numbers of interest and loop over them
# Use old solution to find the new one with advantage
we_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for we_num in we_list:
    delta.assign(0.5 / we_num)
    solver.solve()
    # Save solution
    (v1, v2, v3, p, b11, b12, b22, b13, b23, b33) = w.split(True)  # "True" creates deep copy
    v_rz = as_vector([v1, v3])
    v_rz_projected = project(v_rz, FunctionSpace(mesh, VectorElement("CG", mesh.ufl_cell(), 2)))
    v_rz_projected.rename("v_rz", "v_rz")
    v1file.write(v_rz_projected, we_num)
    v2.rename("v", "v_phi");
    v2file.write(v2, float(we_num))
    p.rename("p", "pressure");
    pfile.write(p, we_num)
