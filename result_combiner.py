import numpy as np
import pyvista as pv
pv.global_theme.allow_empty_mesh = True

def combine_in_out(result_path, inner_result_name, outer_result_name,
                   ss_path, ss_name,
                   plot_slice=True,slice_normal='z',slice_origin=[0,0,0]):
    inner_mesh = pv.read(result_path + inner_result_name + '.vtk')
    outer_mesh = pv.read(result_path + outer_result_name + '.vtk')
    ss_mesh = pv.read(ss_path + ss_name + '.stl')

    # ss_mesh = ss_mesh.decimate(target_reduction=0.3)
    ss_mesh_slice = ss_mesh.slice(normal=slice_normal, origin=slice_origin)
    inner_mesh_slice = inner_mesh.slice(normal=slice_normal, origin=slice_origin)

    ss_mesh_inner = ss_mesh.copy()
    r_ss_mesh_inner = np.linalg.norm(ss_mesh_inner.points, axis=1)

    ss_mesh_inner.points = ss_mesh_inner.points*(0.25/r_ss_mesh_inner[:,None] + 0.75)
    source_inner = ss_mesh_slice.merge(ss_mesh_inner.slice(normal=slice_normal, origin=slice_origin))

    ss_mesh_inner.points = ss_mesh_inner.points*(0.5/r_ss_mesh_inner[:,None] + 0.5)
    source_inner = source_inner.merge(ss_mesh_inner.slice(normal=slice_normal, origin=slice_origin))

    ss_mesh_inner.points = ss_mesh_inner.points*(0.75/r_ss_mesh_inner[:,None] + 0.25)
    source_inner = source_inner.merge(ss_mesh_inner.slice(normal=slice_normal, origin=slice_origin))

    inner_Blines_slice = inner_mesh.streamlines_from_source(source_inner, progress_bar=True, max_time=30)

    merged_Blines = []

    for i_line in range(inner_Blines_slice.n_cells):

        Bline = inner_Blines_slice.extract_cells(i_line)
        Bline_rs = np.linalg.norm(Bline.points, axis=1)
        top_arg = np.nanargmax(Bline_rs)
        top_pt = Bline.points[top_arg]
        top_br = Bline['Br'][top_arg]
        top_bxyz = Bline['Bxyz'][top_arg]
        if abs(top_br) > 1e-2:
            steps = np.arange(-0.1, 0.1, 0.02)
            test_points = np.array([top_pt + step * top_bxyz / np.linalg.norm(top_bxyz) for step in steps])
            test_points_r = np.linalg.norm(test_points,axis=1)
            # print(test_points_r)
            test_points=test_points[test_points_r.argsort(),:]
            # print(test_points)
            test_points_poly = pv.PolyData(test_points)
            selected = test_points_poly.select_enclosed_points(ss_mesh, inside_out=True)
            mask = np.array(selected['SelectedPoints'])
            print(mask)
            # new_line = Bline
            if any(mask == 1):
                ssfp_xyz = test_points[mask == 1][0]
                outer_Bline = outer_mesh.streamlines('Bxyz',start_position=ssfp_xyz, progress_bar=True, max_time=1000)
                outer_Bline['Br'] = outer_Bline['Br'] * np.sign(top_br)
                # n = len(outer_Bline)
                # cells.append((np.hstack([[n], np.arange(offset, offset + n)])))
                Bline_sort_mask = np.argsort(Bline_rs)
                Bline_sort = Bline.points[Bline_sort_mask]

                outer_Bline_rs = np.linalg.norm(outer_Bline.points,axis=1)
                outer_Bline_sort_mask = np.argsort(outer_Bline_rs)
                outer_Bline_sort = outer_Bline.points[outer_Bline_sort_mask]
                new_line_pts = np.vstack([Bline_sort,outer_Bline_sort])
                new_line = pv.lines_from_points(new_line_pts)
                # new_line = pv.PolyData(new_line)
                new_line['Br'] = np.hstack([Bline['Br'][Bline_sort_mask],outer_Bline['Br'][outer_Bline_sort_mask]])

                merged_Blines.append(new_line)
            else:
                merged_Blines.append(Bline)
        else:

            merged_Blines.append(Bline)




    # %%
    if plot_slice:

        p = pv.Plotter()
        for merged_bline in merged_Blines:
            p.add_mesh(merged_bline)
        p.add_mesh(inner_mesh_slice,scalars='Btot',opacity=0.4,cmap='jet',log_scale=True)
        # p.add_mesh(outer_mesh_slice,scalars='Btot',opacity=0.5,cmap='jet',log_scale=True)
        p.show_axes()
        p.show_grid()
        p.show()

    return inner_Blines_slice, merged_Blines, inner_mesh_slice

def combine_in_out_trace_from_photosphere(result_path, inner_result_name, outer_result_name,
                   ss_path, ss_name,theta_resolution=18,phi_resolution=36,
                   plot_slice=True,slice_normal='z',slice_origin=[0,0,0]):
    inner_mesh = pv.read(result_path + inner_result_name + '.vtk')
    outer_mesh = pv.read(result_path + outer_result_name + '.vtk')
    ss_mesh = pv.read(ss_path + ss_name + '.stl')
    sph_mesh = pv.Sphere(radius=1.1,theta_resolution=theta_resolution,phi_resolution=phi_resolution)

    ss_mesh_deci = ss_mesh.decimate_boundary(target_reduction=0.5, progress_bar=True)

    # %%
    inner_mesh['Bxyz'] = inner_mesh['Bxyz']
    outer_mesh['Bxyz'] = outer_mesh['Bxyz']

    inner_Blines_slice = inner_mesh.streamlines_from_source(sph_mesh, progress_bar=True, max_time=10)

    return inner_Blines_slice

def trace_from_photosphere_circle(result_path, inner_result_name, outer_result_name,
                                  ss_path, ss_name,
                                  theta_resolution=18,phi_resolution=36,
                                  normal=[1,0,0],polar=[0,0,1]
                                  ):
    inner_mesh = pv.read(result_path + inner_result_name + '.vtk')
    outer_mesh = pv.read(result_path + outer_result_name + '.vtk')
    ss_mesh = pv.read(ss_path + ss_name + '.stl')
    sph_mesh = pv.Sphere(radius=1.1,theta_resolution=theta_resolution,phi_resolution=phi_resolution)

    ss_mesh_deci = ss_mesh.decimate_boundary(target_reduction=0.5, progress_bar=True)

    seed_circle = pv.CircularArcFromNormal(center=[0., 0., 0., ], resolution=300, normal=normal, polar=polar,
                                                angle=360)
    seed_circle_p = seed_circle.rotate_vector(polar,15,inplace=False)
    seed_circle_m = seed_circle.rotate_vector(polar,-15,inplace=False)
    seed = seed_circle.merge(seed_circle_p)
    seed = seed.merge(seed_circle_m)


    inner_mesh['Bxyz'] = inner_mesh['Bxyz']
    outer_mesh['Bxyz'] = outer_mesh['Bxyz']

    inner_Blines_slice_from_sun = inner_mesh.streamlines_from_source(seed, progress_bar=True, max_time=1000)
    Blines_slice_from_sun = []
    # cells = []
    # offset = 0
    for i_line in range(inner_Blines_slice_from_sun.n_cells):

        Bline = inner_Blines_slice_from_sun.extract_cells(i_line)
        Bline_rs = np.linalg.norm(Bline.points, axis=1)
        top_arg = np.nanargmax(Bline_rs)
        top_pt = Bline.points[top_arg]
        top_br = Bline['Br'][top_arg]
        top_bxyz = Bline['Bxyz'][top_arg]
        if abs(top_br) > 1e-2:
            steps = np.arange(-0.3, 0.3, 0.05)
            test_points = np.array([top_pt + step * top_bxyz / np.linalg.norm(top_bxyz) for step in steps])
            test_points_r = np.linalg.norm(test_points,axis=1)
            # print(test_points_r)
            test_points=test_points[test_points_r.argsort(),:]
            # print(test_points)
            test_points_poly = pv.PolyData(test_points)
            selected = test_points_poly.select_enclosed_points(ss_mesh, inside_out=True)
            mask = np.array(selected['SelectedPoints'])
            print(mask)
            # new_line = Bline
            if any(mask == 1):
                ssfp_xyz = test_points[mask == 1][0]
                outer_Bline = outer_mesh.streamlines('Bxyz',start_position=ssfp_xyz, progress_bar=True, max_time=1000)
                outer_Bline['Br'] = outer_Bline['Br'] * np.sign(top_br)
                # n = len(outer_Bline)
                # cells.append((np.hstack([[n], np.arange(offset, offset + n)])))
                Bline_sort_mask = np.argsort(Bline_rs)
                Bline_sort = Bline.points[Bline_sort_mask]

                outer_Bline_rs = np.linalg.norm(outer_Bline.points,axis=1)
                outer_Bline_sort_mask = np.argsort(outer_Bline_rs)
                outer_Bline_sort = outer_Bline.points[outer_Bline_sort_mask]
                new_line_pts = np.vstack([Bline_sort,outer_Bline_sort])
                new_line = pv.lines_from_points(new_line_pts)
                new_line['Br'] = np.hstack([Bline['Br'][Bline_sort_mask],outer_Bline['Br'][outer_Bline_sort_mask]])

                Blines_slice_from_sun.append(new_line)



    return inner_Blines_slice_from_sun, Blines_slice_from_sun


if __name__ == '__main__':

    combine_in_out(*('RESULT/', '((SphR3Ref2-SphR1Ref2_Ref0)c2158_SS-SphR1Ref2_Ref0)c2158',
                     '(SphR10Ref2-(SphR3Ref2-SphR1Ref2_Ref0)c2158_SS_Ref0)const', 'MESH/2D/',
                     '(SphR3Ref2-SphR1Ref2_Ref0)c2158_SS'),
                   plot_slice=True,slice_normal='x',slice_origin=[0,0,0])
