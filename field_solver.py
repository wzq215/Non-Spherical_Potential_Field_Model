import numpy as np
from datetime import datetime
import pyvista
import ufl
from dolfinx import fem, io, plot
from dolfinx.io.gmshio import read_from_msh
from ufl import ds, dx, grad, dot
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import math
from scipy import interpolate
from scipy import constants
import sunpy.map

# MY MODULE
import utils
from magmap_generator import magmap_generator


def xyz2rlonlat(xyz, for_psi=False):
    r_carrington = (xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2) ** (1 / 2)

    lon_carrington = math.asin(xyz[1] / (xyz[0] ** 2 + xyz[1] ** 2) ** (1 / 2))
    if xyz[0] < 0:
        lon_carrington = np.pi - lon_carrington
    if lon_carrington < 0:
        lon_carrington += 2 * np.pi

    lat_carrington = np.pi / 2 - math.acos(xyz[2] / r_carrington)
    if for_psi:
        lat_carrington = np.pi / 2 - lat_carrington
    return [r_carrington, lon_carrington, lat_carrington]


def calculate_B_field(mesh_u):
    mesh_b = mesh_u.compute_derivative(gradient='Bxyz', divergence=True, vorticity=True, )
    mesh_b.set_active_vectors('Bxyz')
    mesh_b['Bxyz'] = mesh_b['Bxyz']  # Gs
    mesh_b['Btot'] = np.sqrt(mesh_b['Bxyz'][:, 0] ** 2 + mesh_b['Bxyz'][:, 1] ** 2 + mesh_b['Bxyz'][:, 2] ** 2)
    mesh_b['Br'] = (mesh_b.points[:, 0] * mesh_b['Bxyz'][:, 0]
                    + mesh_b.points[:, 1] * mesh_b['Bxyz'][:, 1]
                    + mesh_b.points[:, 2] * mesh_b['Bxyz'][:, 2]
                    ) / np.sqrt(mesh_b.points[:, 0] ** 2 + mesh_b.points[:, 1] ** 2 + mesh_b.points[:, 2] ** 2)
    return mesh_b


def fem_solver(shell_path, shell_name,
               inner_boundary_marker=2, outer_boundary_marker=1,
               magmap_method='dipole', magmap_pathfilename='',
               abs_field=False, magmap_tag='',
               magmap_from='fits', magmap_input=None,magmap_lon_input=None,magmap_lat_input=None,
               result_path='RESULT/',
               **kwargs):

    d1 = datetime.now()

    msh, cell_tags, facet_tags = read_from_msh(shell_path + shell_name + '.msh', MPI.COMM_WORLD, 0, gdim=3)
    inner_boundary = facet_tags.find(inner_boundary_marker)
    outer_boundary = facet_tags.find(outer_boundary_marker)

    # %% 创建函数空间，
    V = fem.FunctionSpace(msh, ('CG', 3))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(msh)

    inner_boundary_dof = fem.locate_dofs_topological(V=V, entity_dim=2, entities=inner_boundary)
    outer_boundary_dof = fem.locate_dofs_topological(V=V, entity_dim=2, entities=outer_boundary)

    phi0 = 0.

    f = fem.Constant(msh, ScalarType(0))

    bc = fem.dirichletbc(value=ScalarType(phi0), dofs=outer_boundary_dof, V=V)

    if magmap_method == 'interp':
        if magmap_from == 'fits':
            magmap = sunpy.map.Map(magmap_pathfilename)
            map_coord = sunpy.map.all_coordinates_from_map(magmap)
            map_lon_ind = np.argsort(map_coord.lon.value,axis=1)
            magmap_lon = np.take_along_axis(map_coord.lon.value,map_lon_ind,axis=1)
            magmap_lat = np.take_along_axis(map_coord.lat.value,map_lon_ind,axis=1)
            magmap_data = np.take_along_axis(magmap.data,map_lon_ind,axis=1)
            f_interp = interpolate.NearestNDInterpolator(list(zip(magmap_lon.ravel(), magmap_lat.ravel())),
                                                         magmap_data.ravel())

        elif magmap_from == 'input':
            magmap_data = magmap_input
            magmap_lon = magmap_lon_input
            magmap_lat = magmap_lat_input
            Lon, Lat = np.meshgrid(magmap_lon, magmap_lat)
            points = np.column_stack((Lon.ravel(), Lat.ravel()))
            f_interp = interpolate.NearestNDInterpolator(points, magmap_data.ravel())

        magmap = fem.Function(V)
        msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
        f_to_c = msh.topology.connectivity(msh.topology.dim - 1, msh.topology.dim)
        msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1)
        c_to_f = msh.topology.connectivity(msh.topology.dim, msh.topology.dim - 1)

        dof_layout = V.dofmap.dof_layout
        coords = V.tabulate_dof_coordinates()
        num_dofs = 0
        for facet in inner_boundary:
            cells = f_to_c.links(facet)
            assert len(cells) == 1
            facets = c_to_f.links(cells[0])
            local_index = np.flatnonzero(facets == facet)
            closure_dofs = dof_layout.entity_closure_dofs(msh.topology.dim - 1, local_index)
            cell_dofs = V.dofmap.cell_dofs(cells[0])
            for dof in closure_dofs:
                local_dof = cell_dofs[dof]
                dof_coordinate = coords[local_dof]
                dof_xyzrlatlon = utils.appendSpherical_np(dof_coordinate.T)
                if dof_xyzrlatlon[5] < 0:
                    dof_xyzrlatlon[5] += np.pi * 2
                dof_Bn = f_interp(np.rad2deg(dof_xyzrlatlon[5]), np.rad2deg(dof_xyzrlatlon[4]))
                # print(local_dof, np.rad2deg(dof_xyzrlatlon[5]), np.rad2deg(dof_xyzrlatlon[4]))
                for b in range(V.dofmap.bs):
                    num_dofs += 1
                    magmap.x.array[local_dof * V.dofmap.bs + b] = dof_Bn
        if abs_field:
            print('ABS IT.')
            magmap = abs(magmap)
    else:
        magmap = magmap_generator(x, method=magmap_method, abs_field=abs_field, **kwargs)



    a = dot(grad(u), grad(v)) * dx
    L = dot(f, v) * dx - dot(magmap, v) * ds

    # Solve
    ksp_type = 'gmres'

    problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={'ksp_type': ksp_type})
    uh = problem.solve()
    print('Solved')
    d2 = datetime.now()
    print(d2 - d1)

    cells, types, x = plot.create_vtk_mesh(V)
    result = pyvista.UnstructuredGrid(cells, types, x)
    result.point_data["u"] = uh.x.array.real

    result = calculate_B_field(result)
    result_name = '(' + shell_name + ')' + magmap_tag

    result.save(result_path + '(' + shell_name + ')' + magmap_tag + '.vtk')
    print('Saving result to: ' + result_path + '(' + shell_name + ')' + magmap_tag + '.vtk')

    return result_path, result_name, result


if __name__ == '__main__':
    # Read MESH
    path = 'MESH/3D/'
    filename = 'SphR3Ref1-SphR1Ref1_Ref1'

    # Extract boundaries
    inner_boundary_marker = 2
    outer_boundary_marker = 1

    result = fem_solver(path, filename,
                        inner_boundary_marker=inner_boundary_marker,
                        outer_boundary_marker=outer_boundary_marker,
                        magmap_method='dipole', abs_field=False, magmap_tag='Dipole',
                        result_path='RESULT/',
                        sph_file='SPHs/mrmqc_c2261.dat', l_max=30,
                        )

    # %%
    result.set_active_scalars('Btot')
    plotter = pyvista.Plotter()
    plotter.add_mesh_slice_orthogonal(result, show_edges=True, opacity=1)
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()
