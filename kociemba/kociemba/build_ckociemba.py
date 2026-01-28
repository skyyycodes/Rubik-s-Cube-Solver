import cffi
import sys

ffi = cffi.FFI()

# Platform-specific compiler flags
if sys.platform == 'win32':
    extra_compile_args = []
else:
    extra_compile_args = ['-std=c99', '-O3', '-D_XOPEN_SOURCE=700']

ffi.set_source(
    "kociemba.ckociembawrapper",
    """
    #include "search.h"

    char* solve(char *cubestring, char *patternstring, char *cache_dir, int max_depth)
    {
        char patternized[64];
        char *sol;

        if (patternstring) {
            patternize(cubestring, patternstring, patternized);
            cubestring = patternized;
        }

        sol = solution(
            cubestring,
            max_depth,
            1000,
            0,
            cache_dir
        );
        return sol;
    }
    """,
    include_dirs=['kociemba/ckociemba/include'],
    sources=[
        'kociemba/ckociemba/prunetable_helpers.c',
        'kociemba/ckociemba/coordcube.c',
        'kociemba/ckociemba/cubiecube.c',
        'kociemba/ckociemba/facecube.c',
        'kociemba/ckociemba/search.c'
    ],
    extra_compile_args=extra_compile_args
)

ffi.cdef("char* solve(char *cubestring, char *patternstring, char *cache_dir, int max_depth);")

if __name__ == "__main__":
    ffi.compile(verbose=True)
