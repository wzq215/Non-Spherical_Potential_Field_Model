"""
run_SPHSS.py

Driver script for running the Potential Field Source Surface model
combined with Schatten Current Sheet model (PFSS+PFCS).

This script executes a simplified NSPF workflow in which the source surface
is prescribed as a spherical shell at a fixed radius, rather than being
extracted self-consistently from the magnetic field magnitude. The magnetic
field at the source surface is represented using spherical harmonic
coefficients and used as the boundary condition for the outer-domain solution.

This workflow is used for:
- Benchmarking against spherical source surface (SphSS) models.
- Producing comparisons presented in Wu et al. (2026), ApJ, submitted.

The script is intended to be executed from the project root directory and
relies on core numerical routines implemented in the `src/` modules.

Author: Ziqi Wu
"""

import os

# ---------------------------------------------------------------------
# Import NSPF core modules
# ---------------------------------------------------------------------
from src.shell_generator import shell_generator
from src.field_solver import fem_solver
from src.sourcesurface_extractor import br_on_ss

# ---------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------
PATH_2D = '../MESH/2D/'
PATH_3D = '../MESH/3D/'
PATH_RESULT = '../RESULT/'
PATH_MAGMAP = '../data/magnetogram/'

# =====================================================================
# ====================== USER INPUT BEGIN =============================
# =====================================================================

# Boundary surface definitions
IB_name = 'SphR1Ref3'        # inner boundary (photosphere)
OB_name = 'SphR2d5Ref3'      # spherical source surface
OOB_name = 'SphR10Ref3'      # outer heliospheric boundary

# Magnetogram selection
magmap_name_list = ['mrzqs240331t2104c2282_050']
magmap_name = magmap_name_list[0]
magmap_tag = magmap_name[-9:]

# Output directory organized by event and snapshot
PATH_RESULT = (
    'RESULT/'
    + 'CR2282_E19/SphSS_Rss1d9_Ref3/'
    + magmap_tag[-3:] + '/'
)

# Source surface parameters
MS_radius = 1.9   # spherical source surface radius [R_sun]
ss_tag = ''       # optional output tag

# =====================================================================
# ======================= USER INPUT END ==============================
# =====================================================================

os.makedirs(PATH_RESULT, exist_ok=True)

# ---------------------------------------------------------------------
# Step 1: Generate spherical shell mesh (photosphere → spherical SS)
# ---------------------------------------------------------------------
shell0_path, shell0_name, readable = shell_generator(
    OB_name, IB_name,
    refine_level=0,
    path_2D=PATH_2D,
    path_3D=PATH_3D
)

# ---------------------------------------------------------------------
# Step 2: Solve PFSS in the inner domain using observed magnetogram
# ---------------------------------------------------------------------
result0_path, result0_name, result0 = fem_solver(
    shell0_path,
    shell0_name,
    magmap_method='interp',
    abs_field=False,
    magmap_pathfilename=PATH_MAGMAP + magmap_name + '.fits.gz',
    magmap_tag=magmap_tag,
    result_path=PATH_RESULT
)

# ---------------------------------------------------------------------
# Step 3: Use the prescribed spherical source surface directly
#         (no SS extraction or geometric smoothing)
# ---------------------------------------------------------------------
ss0_path = PATH_2D
ss0_name = OB_name

shell1_path = shell0_path
shell1_name = shell0_name
result1_path = result0_path
result1_name = result0_name
result1 = result0

# ---------------------------------------------------------------------
# Step 4: Evaluate magnetic field on the spherical SS and compute
#         spherical harmonic representation
# ---------------------------------------------------------------------
ss_xyz_rlatlon, br_ss, clm_ss = br_on_ss(
    result1_name,
    ss0_name,
    l_max=30,
    path_result=result0_path
)

# ---------------------------------------------------------------------
# Step 5: Generate outer shell (SS → outer heliospheric boundary)
# ---------------------------------------------------------------------
shell2_path, shell2_name, readable = shell_generator(
    OOB_name, ss0_name,
    refine_level=0,
    path_2D=PATH_2D,
    path_3D=PATH_3D
)

# ---------------------------------------------------------------------
# Step 6: Solve outer-domain NSPF using spherical-harmonic SS boundary
# ---------------------------------------------------------------------
result2_path, result2_name, result2 = fem_solver(
    shell2_path,
    shell2_name,
    magmap_method='array',
    abs_field=True,
    magmap_tag='array',
    result_path=PATH_RESULT,
    clm=clm_ss,
    l_max=30
)