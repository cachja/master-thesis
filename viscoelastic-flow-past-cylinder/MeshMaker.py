# Mesh maker for the Oldroyd-B_Giesekus_Combo.py script
# generates Meshes No. {1,2,3} with growing DoFs to the ./mesh_file/
# Meshes correspond to the symmetrical part of the classical benchmark flow past the cylinder.
# Ready to be run! Do not run in paralel!

from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import os

# Function for the correction of the curved boundary on the cylinder
def py_snap_boundary(mesh, sub_domain):
    boundary = BoundaryMesh(mesh, "exterior")
    dim = mesh.geometry().dim()
    x = boundary.coordinates()
    for i in range(0, boundary.num_vertices()):
        sub_domain.snap(x[i, :])
    ALE.move(mesh, boundary)


# This function says where to correct the curved boundary after local refinement of the mesh
class Cylinder(SubDomain):
    def snap(self, x):
        r = sqrt((x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2)
        if r <= R:
            x[0] = center[0] + (R / r) * (x[0] - center[0])
            x[1] = center[1] + (R / r) * (x[1] - center[1])

    def inside(self, x, on_boundary):
        r = sqrt((x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2)
        return ((r <= 2.0 * R) and on_boundary)


# Proportions
Len = 40.0
Wid = 2.0
R = 1.0
center = Point(Len / 2, 0)

# Parameters for the mesh generation – gives sequence of Meshes No. {1,2,3}
# [Mesh No., cylinder_refinement, mesh_refinement, local_refinement_iterations]
mesh_params = [[1, 100, 200, 1], [2, 150, 300, 2], [3, 200, 400, 2]]

for params in mesh_params:
    # Mesh refinement parameters
    cylinder_refinement = params[2]
    mesh_refinement = params[1]
    local_refinement_iterations = params[3]
    print(
        f"cylinder_refinement = {cylinder_refinement} \t mesh_refinement = {mesh_refinement} \t local_refinement_iterations = {local_refinement_iterations}")

    # Create mesh
    channel = Rectangle(Point(0, 0), Point(Len, Wid))
    cylinder = Circle(center, R, cylinder_refinement)
    domain = channel - cylinder
    mesh = generate_mesh(domain, mesh_refinement)
    mesh.init()

    # Initialize cylinder class for the mesh correction on the cylinder boundary
    cylinder = Cylinder()

    # Local refinement
    parameters["refinement_algorithm"] = "plaza_with_parent_facets"
    for k in range(local_refinement_iterations):
        info("refinement level {}".format(k))
        cf = MeshFunction('bool', mesh, mesh.topology().dim(), False)
        for c in cells(mesh):
            for vert in vertices(c):
                if (vert.point()[0] > 15 and vert.point()[0] < 25): cf[c] = True

        mesh = refine(mesh, cf, redistribute=False)
        py_snap_boundary(mesh, cylinder)  # Correction of mesh around cylinder
    mesh.init()

    # Write mesh to file
    mesh_file = HDF5File(mesh.mpi_comm(), f"mesh_file/mesh{params[0]}.h5", "w")
    mesh_file.write(mesh, '/mesh')
    print("Mesh's ready and waiting.")

    # Min/max cell diameters
    h_min = MPI.min(mesh.mpi_comm(), mesh.hmin())
    h_max = MPI.max(mesh.mpi_comm(), mesh.hmax())
    print(f"h_min = {h_min} \t h_max = {h_max}")

    # Nice vector plot of mesh cuts off (smaller and larger)
    ## Prepare file
    folder_name = "mesh_pdf"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    ## Plot
    plt.ion()
    MyPlot = plt.figure()
    graf = MyPlot.add_subplot(frameon=False)  # No frame
    graf.set_xlim(18.5, 21.5)
    graf.set_ylim(-0.1, 2.1)
    graf.axes.get_xaxis().set_visible(False)  # No axis
    graf.axes.get_yaxis().set_visible(False)  # No axis

    plot(mesh, linewidth=0.00001)

    plt.savefig(f'mesh_pdf/mesh{params[0]}_1.pdf', dpi=300, format="pdf", bbox_inches="tight")

    graf.set_xlim(10, 30)
    graf.set_ylim(-0.1, 2.1)

    plot(mesh, linewidth=0.00001)

    plt.savefig(f'mesh_pdf/mesh{params[0]}_2.pdf', dpi=300, format="pdf", bbox_inches="tight")
