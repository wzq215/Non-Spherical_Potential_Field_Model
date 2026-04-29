"""
plot_nspf_vs_obs.py

Visualization script for comparing NSPF magnetic field models with
coronal observations from LASCO C2 and SDO/AIA for a single snapshot.

This script performs:
1. Retrieval (or loading from cache) of LASCO C2 coronagraph data and
   SDO/AIA 171 Å EUV images near a given magnetogram time.
2. Coordinate transformations from Carrington to Heliocentric Inertial.
3. Extraction and slicing of NSPF magnetic field solutions.
4. Tracing and merging of magnetic field lines across the source surface.
5. Line-of-sight–aligned visualization of modeled field lines overlaid
   on coronal observations.

This script is intended for analysis and figure production only and is not
part of the core NSPF solver.

Associated publication:
Wu et al. (2026), ApJS, submitted.

Author: Ziqi Wu
"""

# ---------------------------------------------------------------------
# Standard library and third-party imports
# ---------------------------------------------------------------------
import os
import numpy as np
import pyvista as pv
from datetime import datetime, timedelta

import sunpy.map
from sunpy.net import Fido, attrs as a
from sunpy.map.maputils import all_coordinates_from_map
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames

# ---------------------------------------------------------------------
# NSPF post-processing utilities
# ---------------------------------------------------------------------
from src.fieldline_tracer import combine_in_out

pv.global_theme.allow_empty_mesh = True


def convert_to_new_frame(mesh_, map_):
    """
    Transform mesh points from Carrington coordinates to the observation
    frame defined by a SunPy map.

    Parameters
    ----------
    mesh_ : pyvista.PolyData
        Mesh whose points are defined in heliographic Carrington coordinates
        (units of R_sun).
    map_ : sunpy.map.Map
        Observational map defining the target reference frame and observer.

    Returns
    -------
    mesh_ : pyvista.PolyData
        Mesh with points transformed to Heliocentric Inertial coordinates.
    """

    point_coord = SkyCoord(
        mesh_.points * u.R_sun,
        obstime=map_.observer_coordinate.obstime,
        frame='heliographic_carrington',
        observer='sun',
        rsun=map_.coordinate_frame.rsun,
        representation_type='cartesian'
    )

    point_coord_hci = point_coord.transform_to(
        frames.HeliocentricInertial(
            obstime=map_.observer_coordinate.obstime
        )
    )

    mesh_.points[:, 0] = point_coord_hci.cartesian.x.value
    mesh_.points[:, 1] = point_coord_hci.cartesian.y.value
    mesh_.points[:, 2] = point_coord_hci.cartesian.z.value

    return mesh_


def mask_lasco_c2(lasco_map):
    """
    Apply standard inner and outer masks to a LASCO C2 image.

    The inner mask removes the occulting disk and Fresnel diffraction
    region, while the outer mask removes low-signal edge regions.

    Parameters
    ----------
    lasco_map : sunpy.map.Map
        LASCO C2 coronagraph image.

    Returns
    -------
    lasco_map : sunpy.map.Map
        Masked LASCO C2 image.
    occult_colormap : matplotlib colormap
        Colormap with masked pixels shown in black.
    """

    pixel_coords = all_coordinates_from_map(lasco_map)
    solar_center = SkyCoord(
        0 * u.deg, 0 * u.deg, frame=lasco_map.coordinate_frame
    )

    pixel_radii = np.sqrt(
        (pixel_coords.Tx - solar_center.Tx) ** 2 +
        (pixel_coords.Ty - solar_center.Ty) ** 2
    )

    # Inner mask (occulter + diffraction)
    mask_inner = pixel_radii < lasco_map.rsun_obs * 2.4
    # Outer mask (low-signal region)
    mask_outer = pixel_radii > lasco_map.rsun_obs * 6.0

    final_mask = mask_inner | mask_outer
    lasco_map.data[final_mask] = 0.0

    occult_colormap = lasco_map.cmap.copy()
    occult_colormap.set_bad('black')

    return lasco_map, occult_colormap


# ---------------------------------------------------------------------
# Main execution block: single-snapshot analysis and visualization
# ---------------------------------------------------------------------
if __name__ == '__main__':

    # ---------------------------------------------------------------
    # Magnetogram metadata and timestamp
    # ---------------------------------------------------------------
    magmap_name = 'mrzqs240331t2104c2282_050'
    magmap_tag = magmap_name[-9:]
    magmap_dt = datetime.strptime(
        magmap_name[5:16], '%y%m%dt%H%M'
    )

    tag_tmp = magmap_tag
    magmap_tmp = sunpy.map.Map('magnetogram/' + magmap_name)

    # ---------------------------------------------------------------
    # Retrieve LASCO C2 observations near magnetogram time
    # ---------------------------------------------------------------
    time_range = a.Time(
        (magmap_dt - timedelta(minutes=30)).strftime('%Y/%m/%d %H:%M'),
        (magmap_dt + timedelta(minutes=30)).strftime('%Y/%m/%d %H:%M')
    )

    query = Fido.search(
        time_range,
        a.Instrument('LASCO'),
        a.Detector('C2')
    )

    cache_dir = '../data/solar_imgs/'
    os.makedirs(cache_dir, exist_ok=True)

    downloaded_files = Fido.fetch(
        query,
        path=os.path.join(cache_dir, '{file}'),
        download=False,
        overwrite=False
    )

    missing_files = [f for f in downloaded_files if not os.path.exists(f)]
    if missing_files:
        downloaded_files = Fido.fetch(
            query, path=os.path.join(cache_dir, '{file}')
        )

    lasco_c2_map = sunpy.map.Map(downloaded_files[0])
    corona_map, corona_cmap = mask_lasco_c2(lasco_c2_map)

    # ---------------------------------------------------------------
    # Retrieve SDO/AIA 171 Å observations
    # ---------------------------------------------------------------
    time_range = a.Time(
        (magmap_dt - timedelta(seconds=30)).strftime('%Y/%m/%d %H:%M'),
        (magmap_dt + timedelta(seconds=30)).strftime('%Y/%m/%d %H:%M')
    )

    query = Fido.search(
        time_range,
        a.Instrument('AIA'),
        a.Wavelength(171 * u.Angstrom)
    )

    downloaded_files = Fido.fetch(
        query,
        path=os.path.join(cache_dir, '{file}'),
        download=False,
        overwrite=False
    )

    missing_files = [f for f in downloaded_files if not os.path.exists(f)]
    if missing_files:
        downloaded_files = Fido.fetch(
            query, path=os.path.join(cache_dir, '{file}')
        )

    hi_map = sunpy.map.Map(downloaded_files[0])

    # ---------------------------------------------------------------
    # Determine observer line-of-sight direction in Carrington frame
    # ---------------------------------------------------------------
    observer_coord = hi_map.observer_coordinate
    observer_coord.observer = 'sun'
    observer_coord_carr = observer_coord.transform_to(
        frames.HeliographicCarrington
    )

    fov_normal = observer_coord_carr.cartesian.xyz.value

    # ---------------------------------------------------------------
    # NSPF result paths and naming conventions
    # ---------------------------------------------------------------
    PATH_2D = '../MESH/2D/'
    PATH_RESULT = (
        f'RESULT/CR2282_E19/NSPF_Rss2d5_Ref3/{tag_tmp}/'
    )

    OuterSphere_Rs = 10
    MiddleSphere_Rs_str = '2d5'
    InnerSphere_Rs = 1
    CR_tag = 'c2282_' + tag_tmp
    SS_tag = ''
    Ref_tag = 'Ref3'

    NAME_RESULT2 = (
        f'(SphR{OuterSphere_Rs}Ref3-'
        f'(SphR{MiddleSphere_Rs_str}Ref3-'
        f'SphR{InnerSphere_Rs}Ref3_Ref0)'
        f'{CR_tag}_SS{SS_tag}_Ref0)array'
    )

    NAME_RESULT1 = (
        f'((SphR{MiddleSphere_Rs_str}Ref3-'
        f'SphR{InnerSphere_Rs}Ref3_Ref0)'
        f'{CR_tag}_SS{SS_tag}-'
        f'SphR{InnerSphere_Rs}Ref3_Ref0){CR_tag}'
    )

    NAME_SS = (
        f'(SphR{MiddleSphere_Rs_str}Ref3-'
        f'SphR{InnerSphere_Rs}Ref3_Ref0)'
        f'{CR_tag}_SS{SS_tag}'
    )

    # ---------------------------------------------------------------
    # Load, slice, and transform NSPF inner-domain solution
    # ---------------------------------------------------------------
    inner_mesh = pv.read(PATH_RESULT + NAME_RESULT1 + '.vtk')
    inner_mesh_slice = inner_mesh.slice(
        normal=fov_normal, origin=[0, 0, 0]
    )
    inner_mesh_slice_HCI = convert_to_new_frame(
        inner_mesh_slice, corona_map
    )

    # ---------------------------------------------------------------
    # Trace and merge NSPF magnetic field lines across SS
    # ---------------------------------------------------------------
    inner_Blines, outer_Blines, _ = combine_in_out(
        PATH_RESULT,
        NAME_RESULT1,
        NAME_RESULT2,
        PATH_2D,
        NAME_SS,
        slice_normal=fov_normal,
        slice_origin=[0, 0, 0]
    )

    # Transform field lines to observation frame
    inner_Blines_HCI = convert_to_new_frame(inner_Blines, hi_map)
    outer_Blines_HCI = [
        convert_to_new_frame(b, hi_map)
        for b in outer_Blines
    ]

    # ---------------------------------------------------------------
    # Visualization: overlay NSPF field lines on observations
    # ---------------------------------------------------------------
    from sunkit_pyvista import SunpyPlotter

    pv.set_plot_theme(pv.themes.DarkTheme())
    p = SunpyPlotter(window_size=(1700, 1500))

    p.plot_map(
        corona_map,
        clip_interval=[10, 99.7] * u.percent,
        opacity=[0.0, 0.0, 1.0],
        cmap='gray'
    )

    p.plot_map(
        hi_map,
        clip_interval=[10, 99] * u.percent,
        opacity=[0.0, 0.9, 1.0]
    )

    inner_Blines_HCI.set_active_scalars('Br')
    p.plotter.add_mesh(
        inner_Blines_HCI.tube(radius=0.005),
        cmap='coolwarm',
        clim=[-10.0, 10.0],
        opacity=0.3
    )

    for b in outer_Blines_HCI:
        if len(b.points) > 0:
            b.set_active_scalars('Br')
            p.plotter.add_mesh(
                b,
                cmap='coolwarm',
                clim=[-0.1, 0.1],
                opacity=[1.0, 0.3, 1.0],
                show_scalar_bar=False
            )

    p.plotter.show_grid()
    p.plotter.add_title(
        magmap_dt.strftime('%Y/%m/%d %H:%M') + '\nNSPF 2.5 Rs'
    )

    observer_coord_HCI = observer_coord.transform_to(
        frames.HeliocentricInertial
    )
    observer_xyz = observer_coord_HCI.cartesian.xyz.value
    p.plotter.camera_position = observer_xyz / 20.0

    p.show()
    p.plotter.screenshot('NSPF_2.2_zoomin.png')
    p.plotter.close()