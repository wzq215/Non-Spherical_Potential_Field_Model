'''
This script is to generate 3D shell mesh for calculation.
Input:
    inner surface (.stl)
    outer surface (.stl)
    refine level (int, default to 1.)
'''
import os
import gmsh


def create_shell(filepath, IB_filename, OB_filename, exportpath, refine_level=1):
    gmsh.initialize()

    gmsh.merge(filepath + OB_filename + '.stl')

    gmsh.merge(filepath + IB_filename + '.stl')

    gmsh.model.geo.synchronize()
    gmsh.model.geo.add_surface_loop([1])
    gmsh.model.geo.add_surface_loop([2])
    gmsh.model.geo.add_volume([1, 2])
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(3)
    for i in range(refine_level):
        gmsh.model.mesh.refine()
    gmsh.model.mesh.optimize()

    surfs = gmsh.model.getEntities(dim=2)
    gmsh.model.addPhysicalGroup(surfs[0][0], [surfs[0][1]], 1)
    gmsh.model.setPhysicalName(surfs[0][0], 1, 'OUTER BOUNDARY')

    gmsh.model.addPhysicalGroup(surfs[1][0], [surfs[1][1]], 2)
    gmsh.model.setPhysicalName(surfs[1][0], 2, 'INNER BOUNDARY')

    vols = gmsh.model.getEntities(dim=3)
    gmsh.model.addPhysicalGroup(vols[0][0], [vols[0][1]], 11)
    gmsh.model.setPhysicalName(vols[0][0], 1, 'SHELL VOLUME')
    gmsh.model.geo.synchronize()

    exportname = OB_filename + '-' + IB_filename + '_Ref' + str(refine_level)
    gmsh.write(exportpath + exportname + '.msh')
    return exportpath, exportname


def read_test(filepath, filename):
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
    os.makedirs(path_2D, exist_ok=True)
    os.makedirs(path_3D, exist_ok=True)
    exportpath_3D, exportname_3D = create_shell(path_2D, ib_name, ob_name, path_3D, refine_level=refine_level)
    readable = read_test(exportpath_3D, exportname_3D)
    return exportpath_3D, exportname_3D, readable


if __name__ == '__main__':
    IBName = 'SphR1Ref1'
    OBName = 'SphR3Ref1'

    ShellPath, ShellName, readable = shell_generator(OBName, IBName, )
