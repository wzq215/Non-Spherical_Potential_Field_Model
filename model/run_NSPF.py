"""
run_NSPF.py

High-level execution script for the Non-Spherical Potential Field (NSPF)
model using observational magnetogram input.

This script performs the full NSPF workflow for a specific Carrington
rotation and event, including:
1. Generation of spherical shell meshes.
2. Finite-element solution of the scalar potential.
3. Extraction and smoothing of the source surface (SS).
4. Iterative refinement using reconstructed SS boundary conditions.
5. Final extension of the magnetic field to an outer boundary.

This script is intended to reproduce the results presented in:
Wu et al. (2026), ApJ, submitted.

It is designed as a driver script and should be executed from the project
root directory. The underlying numerical methods are implemented in the
`src/` modules.

Author: Ziqi Wu
"""

# ---------------------------------------------------------------------
# Project setup: ensure correct import paths
# ---------------------------------------------------------------------
import os
PROJECT_ROOT = '/Users/ephe/codes/NonSphericalPotentialField/'
os.chdir(PROJECT_ROOT)

import sys
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------
# Import NSPF core modules
# ---------------------------------------------------------------------
from src.shell_generator import shell_generator
from src.field_solver import fem_solver
from src.sourcesurface_extractor import (
    source_surface_extractor,
    br_on_ss,
    br_on_ss_interp,
)

# ---------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------
PATH_2D = 'MESH/2D/'
PATH_3D = 'MESH/3D/'
PATH_RESULT = 'RESULT/'
PATH_MAGMAP = 'data/magnetogram/'

# =====================================================================
# ====================== USER INPUT BEGIN =============================
# =====================================================================

# Inner / source / outer boundary surface names
# See Figure 1 in Wu et al., (2026)
IB_name = 'SphR1Ref3'        # photospheric inner boundary
OB_name = 'SphR2d5Ref3'      # initial source surface boundary
OOB_name = 'SphR10Ref3'      # exit sphere

# Magnetogram selection
magmap_name_list = ['mrzqs240331t2104c2282_050']
magmap_name = magmap_name_list[0]
magmap_tag = magmap_name[-9:]

# Output directory organized by event and snapshot
PATH_RESULT = (
    'RESULT/'
    + 'CR2282_E19/NSPF_Rss2d5_Ref3/'
    + magmap_tag[-3:] + '/'
)

# Source surface parameters
MS_radius = 2.5   # original source surface radius [R_sun]
SS_Br = -1        # auto-detect |B| level for SS extraction, or set the explicit value for isosurface (nT)
ss_tag = ''       # optional tag for SS outputs

# =====================================================================
# ======================= USER INPUT END ==============================
# =====================================================================

os.makedirs(PATH_RESULT, exist_ok=True)

# ---------------------------------------------------------------------
# Step 1: Generate initial shell mesh (photosphere → source surface)
# ---------------------------------------------------------------------
shell0_path, shell0_name, readable = shell_generator(
    OB_name, IB_name,
    refine_level=0,
    path_2D=PATH_2D,
    path_3D=PATH_3D
)

# ---------------------------------------------------------------------
# Step 2: Solve NSPF in inner domain using observed magnetogram
# ---------------------------------------------------------------------
result0_path, result0_name, result0 = fem_solver(
    shell0_path, shell0_name,
    magmap_method='interp',
    abs_field=False,
    magmap_pathfilename=PATH_MAGMAP + magmap_name + '.fits.gz',
    magmap_tag=magmap_tag,
    result_path=PATH_RESULT
)

# ---------------------------------------------------------------------
# Step 3: Extract and smooth the non-spherical source surface (SS)
# ---------------------------------------------------------------------
ss0_path, ss0_name, ss0 = source_surface_extractor(
    result0_path,
    result0_name,
    Rss=MS_radius,
    ss_Btot=SS_Br,
    ss_tag=ss_tag,
    sph_ini_name=OB_name
)

# ---------------------------------------------------------------------
# Step 4: Regenerate inner shell using the reconstructed NSSS
#         and re-solve NSPF
# ---------------------------------------------------------------------
shell1_path, shell1_name, readable = shell_generator(
    ss0_name, IB_name,
    refine_level=0,
    path_2D=PATH_2D,
    path_3D=PATH_3D
)

result1_path, result1_name, result1 = fem_solver(
    shell1_path, shell1_name,
    magmap_method='interp',
    abs_field=False,
    magmap_pathfilename=PATH_MAGMAP + magmap_name + '.fits.gz',
    magmap_tag=magmap_tag,
    result_path=PATH_RESULT
)

# ---------------------------------------------------------------------
# Step 5: Evaluate magnetic field on the SS
# ---------------------------------------------------------------------
ss_xyz_rlatlon, br_ss, clm_ss = br_on_ss(
    result1_name,
    ss0_name,
    l_max=30,
    path_result=result0_path
)

# ---------------------------------------------------------------------
# Step 6: Generate outer shell, from NSSS to the Exit sphere
# ---------------------------------------------------------------------
shell2_path, shell2_name, readable = shell_generator(
    OOB_name, ss0_name,
    refine_level=0,
    path_2D=PATH_2D,
    path_3D=PATH_3D
)

# ---------------------------------------------------------------------
# Step 7: Interpolate SS magnetic field and solve outer NSPF domain
# ---------------------------------------------------------------------
ss_lon, ss_lat, ss_Btot = br_on_ss_interp(
    result1_name,
    ss0_name,
    path_result=result0_path
)

result2_path, result2_name, result2 = fem_solver(
    shell2_path,
    shell2_name,
    magmap_method='interp',
    magmap_from='input',
    result_path=PATH_RESULT,
    abs_field=True,
    magmap_input=abs(ss_Btot),
    magmap_lon_input=ss_lon,
    magmap_lat_input=ss_lat
)