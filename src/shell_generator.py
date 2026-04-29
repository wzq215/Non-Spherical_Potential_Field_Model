"""
shell_generator.py

Utilities for generating spherical shell meshes for the Non-Spherical
Potential Field (NSPF) model using Gmsh.

This module constructs a 3D spherical shell volume from two surface meshes
(inner boundary and outer boundary), assigns physical groups for finite-element
boundary conditions, and exports the resulting mesh in Gmsh (.msh) format.
The generated meshes are intended to be used by dolfinx-based finite-element
solvers.

Typical workflow:
1. Provide STL surface meshes for the inner and outer boundaries.
2. Generate and optionally refine a 3D tetrahedral shell mesh.
3. Assign physical groups for inner boundary, outer boundary, and volume.
4. Verify mesh readability with dolfinx.

This module is intended to be imported and used by higher-level mesh-generation
or preprocessing scripts, and is not designed to be executed as a standalone
program.

Associated publication:
Wu et al. (2026), ApJ, submitted.

Author: Ziqi Wu
"""

import os
import gmsh


def create_shell(filepath, IB_filename, OB_filename, exportpath, refine_level=1):
    """
    Create a 3D spherical shell mesh from inner and outer surface meshes.

    Two STL surface meshes defining the inner and outer boundaries are merged,
    combined into a closed surface loop, and used to generate a tetrahedral
    volume mesh. Physical groups are assigned to the inner boundary, outer
    boundary, and shell volume for later use in finite-element simulations.

    Parameters
    ----------
    filepath : str
        Directory containing the STL surface meshes.
    IB_filename : str
        File name (without extension) of the inner-boundary surface mesh.
    OB_filename : str
        File name (without extension) of the outer-boundary surface mesh.
    exportpath : str
        Directory where the generated 3D mesh will be saved.
    refine_level : int, optional
        Number of uniform mesh-refinement steps applied after initial
        mesh generation.

    Returns
    -------
    exportpath : str
        Path to the directory containing the generated mesh.
    exportname : str
        Base name of the exported mesh file (without extension).
    """

    # Initialize Gmsh and import inner/outer boundary surface meshes
    gmsh.initialize()
    gmsh.merge(filepath + OB_filename + '.stl')
    gmsh.merge(filepath + IB_filename + '.stl')

    # Construct surface loops and define the shell volume
    gmsh.model.geo.synchronize()
    gmsh.model.geo.add_surface_loop([1])
    gmsh.model.geo.add_surface_loop([2])
    gmsh.model.geo.add_volume([1, 2])
    gmsh.model.geo.synchronize()

    # Generate and refine 3D tetrahedral mesh
    gmsh.model.mesh.generate(3)
    for i in range(refine_level):
        gmsh.model.mesh.refine()
    gmsh.model.mesh.optimize()

    # Assign physical groups for FEM boundary conditions
    surfs = gmsh.model.getEntities(dim=2)
    # Outer boundary surface
    gmsh.model.addPhysicalGroup(surfs[0][0], [surfs[0][1]], 1)
    gmsh.model.setPhysicalName(surfs[0][0], 1, 'OUTER BOUNDARY')
    # Inner boundary surface
    gmsh.model.addPhysicalGroup(surfs[1][0], [surfs[1][1]], 2)
    gmsh.model.setPhysicalName(surfs[1][0], 2, 'INNER BOUNDARY')
    # Shell Volume
    vols = gmsh.model.getEntities(dim=3)
    gmsh.model.addPhysicalGroup(vols[0][0], [vols[0][1]], 11)
    gmsh.model.setPhysicalName(vols[0][0], 1, 'SHELL VOLUME')

    gmsh.model.geo.synchronize()

    # Export the mesh in Gmsh format
    exportname = OB_filename + '-' + IB_filename + '_Ref' + str(refine_level)
    gmsh.write(exportpath + exportname + '.msh')
    return exportpath, exportname


def read_test(filepath, filename):
    """
    Test whether a generated Gmsh mesh can be successfully read by dolfinx.

    This function attempts to load the mesh using dolfinx's Gmsh I/O interface,
    serving as a basic validity check for downstream finite-element solvers.

    Parameters
    ----------
    filepath : str
        Directory containing the mesh file.
    filename : str
        Base name (without extension) of the mesh file.

    Returns
    -------
    readable : bool
        True if the mesh can be read successfully, False otherwise.
    """

    from mpi4py import MPI
    model_rank = 0
    from dolfinx.io.gmshio import read_from_msh
    try:
        msh, cell_tags, facet_tags = read_from_msh(filepath + filename + '.msh', MPI.COMM_WORLD, model_rank, gdim=3)
        return True
    except:
        print('Not Readable :(')
        return False


def shell_generator(ob_name, ib_name, refine_level=1,
                    path_2D='MESH/2D/', path_3D='MESH/3D/'):
    """
    High-level interface for generating and validating a spherical shell mesh.

    This function orchestrates the creation of a 3D shell mesh from 2D surface
    meshes, ensures output directories exist, and performs a basic readability
    test using dolfinx.

    Parameters
    ----------
    ob_name : str
        File name (without extension) of the outer-boundary surface mesh.
    ib_name : str
        File name (without extension) of the inner-boundary surface mesh.
    refine_level : int, optional
        Number of mesh-refinement steps.
    path_2D : str, optional
        Directory containing the 2D STL surface meshes.
    path_3D : str, optional
        Directory where the generated 3D mesh will be stored.

    Returns
    -------
    exportpath_3D : str
        Path to the generated 3D mesh file.
    exportname_3D : str
        Base name of the generated mesh.
    readable : bool
        Whether the generated mesh is readable by dolfinx.
    """

    os.makedirs(path_2D, exist_ok=True)
    os.makedirs(path_3D, exist_ok=True)
    exportpath_3D, exportname_3D = create_shell(path_2D, ib_name, ob_name, path_3D, refine_level=refine_level)
    readable = read_test(exportpath_3D, exportname_3D)
    return exportpath_3D, exportname_3D, readable


