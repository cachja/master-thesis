# This script solves the classical benchmark of flow past the cylinder. It uses the model
# of the convex combination of Oldroyd-B with Giesekus model with stress diffusion.
# It runs over list of Weissenberg numbers w_list[] for each beta from beta_list[],
# and it provides two possibilities how to do it:
## Simple_We_step() just runs over w_list with basic homotopy (continuation) method, i.e.,
### the solution of preceedig We is an estimate for the solution with new We
## Iterative_We() do the same thing as the Simple_We_step(), except if the solution with the
### new We do not converge, it solves auxiliary problem with We_help = (We_old+We_new)/2.0,
### up to the depth 5 of such auxiliary steps (then it breaks out of the loop)

# It enables solve problem even for We = 200 if the minimal lambda is known. Now, there is
# used an extrapolation of minimal lambda for beta = 1.

# Note that computations for all beta and for a lot of We is time demanding on fine meshes.

# Ready to be used, if the meshes are generated in the ./mesh_file/ by MeshMaker

from dolfin import *
from mshr import *
import signal
import numpy as np
import os


# Just a thing to make sure that data will print out if the program is stopped by hand
def keyboard_interrupt_handler(signal, frame):
    print("Odchycen KeyboardInterrupt, zaviram soubor.")
    drag_file.close()
    exit(1)
signal.signal(signal.SIGINT, keyboard_interrupt_handler);

PETScOptions.set('mat_mumps_icntl_14', 1000)  # work array, multiple of estimate to allocate
PETScOptions.set('mat_mumps_icntl_24', 1)  # detect null pivots
PETScOptions.set('mat_mumps_cntl_1', 1.0)  # pivoting threshold, this solves to machine precision
PETScOptions.set("snes_divergence_tolerance", 1e2) # divergence tolerance of the SNES, better to do it on your own than let residuum be huge which may trigger an exception

comm = MPI.comm_world
rank = MPI.rank(comm)
set_log_level(LogLevel.INFO if rank == 0 else LogLevel.INFO)
parameters["std_out_all_processes"] = False
parameters["form_compiler"]["quadrature_degree"] = 8
parameters["refinement_algorithm"] = "plaza_with_parent_facets"
parameters["ghost_mode"] = "shared_facet"

# Create mesh
## Proportions
L = 40.0
Wid = 2.0
R = 1.0
center = Point(L / 2, 0)

# Load Mesh
mesh = Mesh()
mesh_name = "mesh1"
hdf = HDF5File(mesh.mpi_comm(), f"./mesh_file/{mesh_name}.h5", "r")
# info(hdf.parameters, True)
hdf.read(mesh, "/mesh", False)
mesh.init()
if rank == 0: print("Mesh loaded")

## Open file for results
folder_name = f"results_{mesh_name}_OGcombo"
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

# Prepare boundaries
boundary_parts = MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)

# Define finite elements
Ev = VectorElement("CG", mesh.ufl_cell(), 2)  # Velocity
EB = FiniteElement("CG", mesh.ufl_cell(), 1)  # Left Cauchy-Green tensor
Ep = FiniteElement("CG", mesh.ufl_cell(), 1)  # Pressure

# Coupling I - Build function spaces (Taylor-Hood) for coupled problem
W = FunctionSpace(mesh, MixedElement([Ev, EB, EB, EB, Ep]))  # order of elements matters

# Define boundaries
for f in facets(mesh):
    mp = f.midpoint()
    if near(mp[0], 0.0):  # inflow
        boundary_parts[f] = 1
    elif near(mp[0], L):  # outflow
        boundary_parts[f] = 2
    elif near(mp[1], Wid):  # upper wall
        boundary_parts[f] = 3
    elif near(mp[1], 0.0):  # wall of symmetry
        boundary_parts[f] = 4
    elif mp.distance(center) <= R + 0.0001:  # cylinder
        boundary_parts[f] = 5

## Facet normal and boundary measure
n = FacetNormal(mesh)
ds = Measure("ds", subdomain_data=boundary_parts)
h = CellDiameter(mesh)

# Define boundary conditions
## velocity NO SLIP boundary condition on upper wall
bc_upper_wall = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundary_parts, 3)
## velocity FREE SLIP boundary condition on wall of symetry
bc_symmetry_wall = DirichletBC(W.sub(0).sub(1), Constant(0.0), boundary_parts, 4)
## velocity NO SLIP boundary condition on cylinder
bc_cylinder = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundary_parts, 5)
## inflow boundary
inflow_profile = Expression((f"3.0/8.0*(4.0-pow(x[1],2))", '0.0'), degree=2)
bc_inflow = DirichletBC(W.sub(0), inflow_profile, boundary_parts, 1)

# Coupling II - we combine boundary conditions for velocity and cauchy tensor
bcs = [bc_upper_wall, bc_symmetry_wall, bc_cylinder, bc_inflow]

# Coupling III - W is a function of velocity, left Cauchy-Green tensor and pressure from W space
v_, b11_, b12_, b22_, p_ = TestFunctions(W)
w = Function(W)
v, b11, b12, b22, p = split(w)

B_ = as_tensor([[b11_, b12_], [b12_, b22_]])
B = as_tensor([[b11, b12], [b12, b22]])

# Previous time step
w0 = Function(W)
(v0, b110, b120, b220, p0) = split(w0)
B0 = as_tensor([[b110, b120], [b120, b220]])

# Initial data
w0ic = Expression(("0.0", "0.0", "1.0", "0.0", "1.0", "0.0"), degree=1)
w0.assign(interpolate(w0ic, W))
w.assign(interpolate(w0ic, W))

# Stabilization over the edges - use only with lambd = 0
alpha = Constant(0.0)
r = alpha * avg(h) ** 2 * inner(jump(grad(B), n), jump(grad(B_), n)) * dS

###### New equations - Giesekus-Oldroyd-B variant
I = Identity(mesh.geometry().dim())
L = grad(v)
D = 0.5 * (L + L.T)
Wv = 0.5 * (L - L.T)


def objective_derivative(a):
    return B.dx(0) * v[0] + B.dx(1) * v[1] - a * (D * B + B * D) - (Wv * B - B * Wv)


# Benchmark parameters
nu = Constant(0.59)
We = Constant(0.1)
G = Constant(0.41 / We)
lambd = Constant(10.0)
beta = Constant(0.0)  # choose 0.0, 0.1, 0.2 in the loop bellow
a = Constant(1.0)  # gives Oldroyd upper convective derivative
# a = Constant(0.0) # gives corrotational Jaumann-Zaremba derivative
delta1 = Constant(1.0 / We)
delta2 = Constant(0.0)

# Define variational problem
T = -p * I + 2.0 * nu * D + G * ((1 - beta) * (B - I) + beta * (B - I) * B)

Eq1 = div(v) * p_ * dx
Eq2 = inner(T, grad(v_)) * dx
Eq3 = inner(objective_derivative(a), B_) * dx + delta1 * inner(B - I, B_) * dx + delta2 * inner((B - I) * B,B_) * dx +\
      lambd * inner(grad(B), grad(B_)) * dx

# Coupling VI - we sum continuity and momentum eqs. and solve Oldroyd-B problem together for pvb function and pvb boundary cond.
Eq = Eq1 + Eq2 + Eq3
# Eq = Eq + r

## prepare solver
info("Solving problem of size: {0:d}".format(W.dim()))
problem = NonlinearVariationalProblem(Eq, w, bcs, derivative(Eq, w))
solver = NonlinearVariationalSolver(problem)
## set solvers parameters
# SNES solver is better for minimizing of lambda as Newton is more likely to trigger error when non-convergence is huge
solver.parameters['nonlinear_solver'] = 'snes'
solver.parameters['snes_solver']['line_search'] = 'cp'
solver.parameters['snes_solver']['linear_solver'] = 'mumps'
solver.parameters['snes_solver']['maximum_iterations'] = 10
solver.parameters['snes_solver']['error_on_nonconvergence'] = False
solver.parameters['snes_solver']['absolute_tolerance'] = 1e-8
solver.parameters['snes_solver']['relative_tolerance'] = 1e-10

# def evaluating functions
def drag_function():
    w_ = Function(W)
    DirichletBC(W.sub(0), (1.0, 0.0), boundary_parts, 5).apply(w_.vector())
    drag = assemble(action(Eq, w_))
    return -2 * drag


def lift_function():
    w_ = Function(W)
    DirichletBC(W.sub(0), (0.0, 1.0), boundary_parts, 5).apply(w_.vector())
    lift = assemble(action(Eq, w_))
    return -2 * lift


def eig_plus(A): return (tr(A) + sqrt(tr(A) ** 2 - 4 * det(A))) / 2


def eig_minus(A): return (tr(A) - sqrt(tr(A) ** 2 - 4 * det(A))) / 2


## Define functions for the main part

# This function runs over w_list with basic homotopy (continuation) method
def Simple_We_step(we_num):
    We.assign(we_num)
    G.assign(0.41 / we_num)
    delta1.assign(1.0 / we_num)
    delta2.assign(0.0)
    lambd.assign(0.1*we_num+0.1) #fit for beta = 1.0
    solver.solve()

# This function runs over w_list with basic homotopy (continuation) method, except if the
# solution do not converge, it solves auxiliary problem with We_help = (We_old+We_new)/2.0
should_break = False
def iterative_We(we_num, we_num0):
    should_break = False
    converged = False  # BUNO - Suppose that solver wont be able solve such big step in We
    w2.assign(w)  # Save last converged solution
    # Loop - if solver in important We converged then we go throught loop just once
    # Idea: if We step is too big and solver diverges then make just half We step
    we_points = []  # Store half steps points in order to go back in these points to important We point
    while True:
        if rank == 0: print(
            f"we_num = {we_num}, we_num0 = {we_num0}, depth = {len(we_points)}, alpha = {float(alpha)}")  # Output
        if len(we_points) == 5:
            should_break = True
            break  # Make sure that loop wont go in cycles and if so then also end for loop above
        We.assign(we_num)
        G.assign(0.41 / we_num)
        delta1.assign(1.0 / we_num)
        delta2.assign(1.0 / we_num)
        lambd.assign(0.1*we_num+0.1) #fit for beta = 1.0
        solver.parameters['newton_solver']['relaxation_parameter'] = 1.0 - len(we_points) * 0.05  # If solver diverges, make him better condition in next time computation
        solver.parameters['newton_solver']['maximum_iterations'] = 10 + len(we_points)
        pair1, converged = solver.solve()  # Returns number of iterations and boolean about convergence + solution w

        if not converged:
            we_points.append(we_num)  # add to list We where solver diverges
            w.assign(w2)  # return to initial solution last converged solution
            tmp = we_num0  # temporary variable
            we_num0 = we_num  # actual We becomes previous We
            we_num -= abs(we_num - tmp) / 2.0  # make only half We step
        else:
            if not we_points:  # if there are no storaged We points, then we solved imporant We and we can continue out of loop
                break
            we_num0 = we_num  # actual We becomes previous We
            we_num = we_points.pop()  # if there are storaged We points, then we take last added We
            w2.assign(w)  # save actual best solution

    solver.parameters['newton_solver']['relaxation_parameter'] = 1.0  # reset solver
    solver.parameters['newton_solver']['maximum_iterations'] = 10
    we_num0 = we_num  # previous We

### ### ### ### ### ###  Main part ### ### ### ### ### ###

# Prepare for loop over all We's
w_list = []  # Important We values, which will be primary evaluated
for k in range(1,10,1):
    w_list.append(k/10.0)
for k in range(1,202,2):
    w_list.append(k)
if rank == 0: print(w_list)

beta_list = []
for k in range(0, 11, 1):
    beta_list.append(k/10.0)
if rank == 0: print(beta_list)

for beta_num in beta_list:
    beta.assign(beta_num)
    info("beta = " + str(float(beta)) + "\n")
    we_num0 = 0.0  # inicialize We in the last step
    w2 = Function(W)  # inicialize save of last solution
    w0.assign(interpolate(w0ic, W))  # Reset functions
    w.assign(interpolate(w0ic, W))  # Reset functions

    ## Open file for results
    drag_file = open(f"{folder_name}/data_beta-{float(beta)}.txt", "w")
    # Save solution in VTK format
    pfile = XDMFFile(comm, f"{folder_name}/p_beta-{float(beta)}.xdmf")
    vfile = XDMFFile(comm, f"{folder_name}/v_beta-{float(beta)}.xdmf")

    for we_num in w_list:
        # Simple_We_step(we_num)
        iterative_We(we_num, we_num0)
        we_num0 = we_num

        # Get results
        drag = drag_function()
        lift = lift_function()
        info("We = " + str(float(We)) + "\t" + "drag = " + str(drag) + "\t" + "lift = " + str(lift) + "\n")

        detB_projected = project(det(B), FunctionSpace(mesh, FiniteElement("DG", mesh.ufl_cell(), 0)))
        min_detB = MPI.min(mesh.mpi_comm(), np.min(detB_projected.vector().get_local()))

        em = project(eig_minus(B), FunctionSpace(mesh, FiniteElement("DG", mesh.ufl_cell(), 0)))
        ep = project(eig_plus(B), FunctionSpace(mesh, FiniteElement("DG", mesh.ufl_cell(), 0)))
        min_eig_minus = MPI.min(mesh.mpi_comm(), np.min(em.vector().get_local()))
        min_eig_plus = MPI.min(mesh.mpi_comm(), np.min(ep.vector().get_local()))

        drag_file.write("We = " + str(float(We)) + "\t" + "drag = " + str(drag) + "\t" + "lift = " + str(
            lift) + "\t" + "lambda = " + str(float(lambd)) + "\t" + "min_detB = " + str(
            float(min_detB)) + "\t" + "min_eig_minus = " + str(float(min_eig_minus)) + "\t" + "min_eig_plus = " + str(
            float(min_eig_plus)) + "\n")
        drag_file.flush()
        v, b11, b12, b22, p = w.split(True)
        pfile.write(p, we_num)
        vfile.write(v, we_num)

        if should_break: break
    drag_file.close()
