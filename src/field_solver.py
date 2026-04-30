"""
field_solver.py

Finite-element solver for the Non-Spherical Potential Field (NSPF) model.

This module solves the Laplace equation for the magnetic scalar potential
in a spherical shell domain using dolfinx, with inner boundary conditions
prescribed by a photospheric magnetogram or analytic magnetic field models.

The resulting magnetic field is obtained as B = -∇Φ and exported in VTK format.

This code is used in:
Wu et al. (2026), ApJS, submitted.

Author: Ziqi Wu
"""

import numpy as np
from datetime import datetime
import pyvista
import ufl
from dolfinx import fem, plot
from dolfinx.io.gmshio import read_from_msh
from ufl import ds, dx, grad, dot
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import math
from scipy import interpolate
import sunpy.map
import sympy as sp
from ufl import sqrt, sin, cos

# MY MODULE
from src.utils import *

def magmap_from_array(x, clm=[], l_max=30, **kwargs):
    """
    Construct the inner-boundary magnetic field from spherical harmonic
    coefficients.

    This function reconstructs the radial magnetic field on the inner
    boundary using spherical harmonic coefficients (g_lm, h_lm) provided
    in `clm`. The resulting field is expressed symbolically and evaluated
    as a UFL expression for use in the finite-element weak formulation.

    Notes
    -----
    - This implementation uses symbolic operations (SymPy) combined with
      UFL expressions and is not optimized for performance.
    - It is intended for low spherical harmonic degrees (l_max ≲ 30),
      validation, and comparison studies.
    - This method should not be used for large-scale production runs due
      to its high computational cost.

    Parameters
    ----------
    x : ufl.SpatialCoordinate
        Spatial coordinate in the finite-element domain.
    clm : ndarray
        Spherical harmonic coefficients with shape (2, l_max+1, l_max+1),
        where clm[0, l, m] = g_lm and clm[1, l, m] = h_lm.
    l_max : int, optional
        Maximum spherical harmonic degree used in the reconstruction.
    **kwargs :
        Reserved for future extensions.

    Returns
    -------
    magmap : ufl.Expression
        Symbolic UFL expression representing the radial magnetic field
        on the inner boundary.
    """

    print('Generating magmap from input sph_array. Lmax=' + str(l_max))

    # Load spherical harmonic coefficients
    # g_lm (cosine terms) and h_lm (sine terms)
    glm_array = clm[0, :, :]
    hlm_array = clm[1, :, :]

    # Define colatitude cosine and longitude in Carrington coordinates
    cos_colat = x[2] / (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** (1 / 2)

    # Longitude definition ensures full [0, 2π) coverage
    lon = (
        ufl.acos(x[0] / ufl.sqrt(x[0] ** 2 + x[1] ** 2))
        * ufl.sign(x[1])
        + ufl.pi
        - ufl.sign(x[1]) * ufl.pi
    )

    # Symbolic variables for spherical harmonic expansion
    COSCOLAT, LON = sp.symbols('cos_colat, lon')

    # Initialize symbolic magnetic field expression
    Br = sp.symbols('0.')

    # Spherical harmonic reconstruction of Br
    # Br(θ, φ) = Σ_l Σ_m P_l^m(cosθ)
    #             [ g_lm cos(mφ) + h_lm sin(mφ) ]
    for i_l in range(l_max + 1):
        for i_m in range(i_l + 1):

            print('l: ', i_l, '; m:', i_m)

            # Associated Legendre function with Schmidt normalization
            Plm = sp.N(
                sp.assoc_legendre(i_l, i_m, COSCOLAT)
                * (2 * math.factorial(i_l - i_m)
                   / math.factorial(i_l + i_m)) ** (1 / 2)
            )

            Br = sp.N(
                Br
                + Plm
                * (
                    glm_array[i_l, i_m] * sp.cos(i_m * LON)
                    + hlm_array[i_l, i_m] * sp.sin(i_m * LON)
                )
            )

    # Convert symbolic expression to UFL-compatible expression
    magmap = eval(str(Br))

    return magmap


def calculate_B_field(mesh_u):
    """
    Compute the magnetic field from the scalar potential.

    The magnetic field is defined as:
        B = -∇Φ

    Parameters
    ----------
    mesh_u : pyvista.UnstructuredGrid
        VTK mesh containing the scalar potential solution Φ.

    Returns
    -------
    mesh_b : pyvista.UnstructuredGrid
        Mesh with magnetic field components:
        - Bxyz : Cartesian components of the magnetic field
        - Br   : Radial component
        - Btot : Magnetic field magnitude
    """

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
               magmap_method='dipole', magmap_pathfilename='',
               abs_field=False, magmap_tag='',
               magmap_from='fits', magmap_input=None,
               magmap_lon_input=None,magmap_lat_input=None,
               result_path='RESULT/',
               **kwargs):
    """
    Solve the NSPF scalar potential using the finite-element method.

    The scalar potential Φ is obtained by solving the Laplace equation
    in a spherical shell domain. The inner boundary condition is prescribed
    by a photospheric magnetogram or an analytic magnetic field model,
    while the outer boundary is fixed to zero potential.

    Parameters
    ----------
    shell_path : str
        Path to the Gmsh mesh file.
    shell_name : str
        Name of the spherical shell mesh (without extension).
    magmap_method : str
        Method used to construct the boundary magnetic field.
        Options include 'interp' and analytic models (e.g., 'dipole').
    magmap_from : str
        Source of the magnetogram ('fits' or 'input').
    result_path : str
        Directory where the VTK result file will be stored.

    Returns
    -------
    result_path : str
        Path to the output directory.
    result_name : str
        Name of the output VTK file.
    result : pyvista.UnstructuredGrid
        VTK mesh containing Φ and derived magnetic field components.
    """

    d1 = datetime.now()

    # --- Load Gmsh mesh file ---
    msh, cell_tags, facet_tags = read_from_msh(shell_path + shell_name + '.msh',
                                               MPI.COMM_WORLD, 0, gdim=3)
    inner_boundary_marker = 2
    outer_boundary_marker = 1
    inner_boundary = facet_tags.find(inner_boundary_marker)
    outer_boundary = facet_tags.find(outer_boundary_marker)

    # %%
    # --- Finite-element space and boundary conditions ---
    # Continuous Galerkin elements of degree 3 are used
    # to solve the Laplace equation for the scalar potential Φ.
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
                dof_xyzrlatlon = appendSpherical_np(dof_coordinate.T)
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
    elif magmap_method == 'dipole':
        b0 = 100.
        print('Generating Dipole field with B0=' + str(b0) + '(nT)')
        magmap = -b0 / (4 * np.pi * 1e-7) * x[2] / (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** (1 / 2)
    elif magmap_method == 'array':
        magmap = magmap_from_array(x, abs_field=abs_field, **kwargs)

    # --- Weak form of the Laplace equation ---
    # ∫ ∇Φ · ∇v dΩ = - ∫ B_n v dS
    # See Equations (1)-(3) in Wu et al., 2026
    a = dot(grad(u), grad(v)) * dx
    L = dot(f, v) * dx - dot(magmap, v) * ds

    # Solve
    ksp_type = 'gmres'
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={'ksp_type': ksp_type})
    uh = problem.solve()
    d2 = datetime.now()
    print('Solved! Time used: ',d2 - d1)

    cells, types, x = plot.create_vtk_mesh(V)
    result = pyvista.UnstructuredGrid(cells, types, x)
    result.point_data["u"] = uh.x.array.real

    result = calculate_B_field(result)
    result_name = '(' + shell_name + ')' + magmap_tag

    result.save(result_path + '(' + shell_name + ')' + magmap_tag + '.vtk')
    print('Saving result to: ' + result_path + '(' + shell_name + ')' + magmap_tag + '.vtk')

    return result_path, result_name, result

