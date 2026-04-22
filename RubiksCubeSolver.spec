# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for cube3d.py — Rubik's Cube interactive 3D solver.
# Build: `pyinstaller RubiksCubeSolver.spec` (run from repo root).
#   Windows → dist/RubiksCubeSolver.exe  (single file)
#   macOS   → dist/RubiksCubeSolver.app  (bundle)

import sys

IS_MAC = sys.platform == 'darwin'
KPKG = 'kociemba/kociemba'

datas = [
    (f'{KPKG}/cprunetables',                   'kociemba/cprunetables'),
    (f'{KPKG}/pykociemba/prunetables',         'kociemba/pykociemba/prunetables'),
    (f'{KPKG}/neural_model.pkl',               'kociemba'),
    (f'{KPKG}/neural_value_model.pkl',         'kociemba'),
    (f'{KPKG}/neural_model_coord.pkl',         'kociemba'),
    (f'{KPKG}/neural_model_scalar.pkl',        'kociemba'),
    (f'{KPKG}/neural_value_model_coord.pkl',   'kociemba'),
    (f'{KPKG}/neural_value_model_scalar.pkl',  'kociemba'),
]

hiddenimports = [
    'kociemba',
    'kociemba.pykociemba',
    'kociemba.pykociemba.search',
    'kociemba.pykociemba.coordcube',
    'kociemba.pykociemba.cubiecube',
    'kociemba.pykociemba.facecube',
    'kociemba.pykociemba.facelet',
    'kociemba.pykociemba.corner',
    'kociemba.pykociemba.edge',
    'kociemba.pykociemba.tools',
    'kociemba.pykociemba.symmetry',
    'kociemba.pykociemba.color',
    'kociemba.neural_heuristic',
    'cffi',
    '_cffi_backend',
]

a = Analysis(
    ['cube3d.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['cv2', 'streamlit', 'torch', 'tensorflow', 'PIL.ImageTk'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

if IS_MAC:
    # onedir + BUNDLE → .app (Apple's recommended pattern).
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='RubiksCubeSolver',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name='RubiksCubeSolver',
    )
    app = BUNDLE(
        coll,
        name='RubiksCubeSolver.app',
        icon=None,
        bundle_identifier='com.rubikssolver.cube3d',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'NSPrincipalClass': 'NSApplication',
        },
    )
else:
    # Windows/Linux → single-file executable.
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='RubiksCubeSolver',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
