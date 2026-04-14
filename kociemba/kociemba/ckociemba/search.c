#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "search.h"
#include "color.h"
#include "facecube.h"
#include "coordcube.h"

#define MIN(a, b) (((a)<(b))?(a):(b))
#define MAX(a, b) (((a)>(b))?(a):(b))

long g_phase1_nodes = 0;

char* solutionToString(search_t* search, int length, int depthPhase1)
{
    char* s = (char*) calloc(length * 3 + 5, 1);
    int cur = 0, i;
    for (i = 0; i < length; i++) {
        switch (search->ax[i]) {
        case 0:
            s[cur++] = 'U';
            break;
        case 1:
            s[cur++] = 'R';
            break;
        case 2:
            s[cur++] = 'F';
            break;
        case 3:
            s[cur++] = 'D';
            break;
        case 4:
            s[cur++] = 'L';
            break;
        case 5:
            s[cur++] = 'B';
            break;
        }
        switch (search->po[i]) {
        case 1:
            s[cur++] = ' ';
            break;
        case 2:
            s[cur++] = '2';
            s[cur++] = ' ';
            break;
        case 3:
            s[cur++] = '\'';
            s[cur++] = ' ';
            break;
        }
        if (i == depthPhase1 - 1) {
            s[cur++] = '.';
            s[cur++] = ' ';
        }
    }
    return s;
}


char* solution(char* facelets, int maxDepth, long timeOut, int useSeparator, const char* cache_dir)
{
    search_t* search = (search_t*) calloc(1, sizeof(search_t));
    facecube_t* fc;
    cubiecube_t* cc;
    coordcube_t* c;

    int s, i;
    int mv, n;
    int busy;
    int depthPhase1;
    time_t tStart;
    // +++++++++++++++++++++check for wrong input +++++++++++++++++++++++++++++
    int count[6] = {0};

    if (PRUNING_INITED == 0) {
        initPruning(cache_dir);
    }

    for (i = 0; i < 54; i++)
        switch(facelets[i]) {
            case 'U':
                count[U]++;
                break;
            case 'R':
                count[R]++;
                break;
            case 'F':
                count[F]++;
                break;
            case 'D':
                count[D]++;
                break;
            case 'L':
                count[L]++;
                break;
            case 'B':
                count[B]++;
                break;
        }

    for (i = 0; i < 6; i++)
        if (count[i] != 9) {
            free(search);
            return NULL;
        }

    fc = get_facecube_fromstring(facelets);
    cc = toCubieCube(fc);
    if ((s = verify(cc)) != 0) {
        free(search);
        return NULL;
    }

    // +++++++++++++++++++++++ initialization +++++++++++++++++++++++++++++++++
    c = get_coordcube(cc);

    search->po[0] = 0;
    search->ax[0] = 0;
    search->flip[0] = c->flip;
    search->twist[0] = c->twist;
    search->parity[0] = c->parity;
    search->slice[0] = c->FRtoBR / 24;
    search->URFtoDLF[0] = c->URFtoDLF;
    search->FRtoBR[0] = c->FRtoBR;
    search->URtoUL[0] = c->URtoUL;
    search->UBtoDF[0] = c->UBtoDF;

    search->minDistPhase1[1] = 1;// else failure for depth=1, n=0
    mv = 0;
    n = 0;
    busy = 0;
    depthPhase1 = 1;

    g_phase1_nodes = 0;
    tStart = time(NULL);

    // +++++++++++++++++++ Main loop ++++++++++++++++++++++++++++++++++++++++++
    do {
        do {
            if ((depthPhase1 - n > search->minDistPhase1[n + 1]) && !busy) {

                if (search->ax[n] == 0 || search->ax[n] == 3)// Initialize next move
                    search->ax[++n] = 1;
                else
                    search->ax[++n] = 0;
                search->po[n] = 1;
            } else if (++search->po[n] > 3) {
                do {// increment axis
                    if (++search->ax[n] > 5) {

                        if (time(NULL) - tStart > timeOut)
                            return NULL;

                        if (n == 0) {
                            if (depthPhase1 >= maxDepth)
                                return NULL;
                            else {
                                depthPhase1++;
                                search->ax[n] = 0;
                                search->po[n] = 1;
                                busy = 0;
                                break;
                            }
                        } else {
                            n--;
                            busy = 1;
                            break;
                        }

                    } else {
                        search->po[n] = 1;
                        busy = 0;
                    }
                } while (n != 0 && (search->ax[n - 1] == search->ax[n] || search->ax[n - 1] - 3 == search->ax[n]));
            } else
                busy = 0;
        } while (busy);

        // +++++++++++++ compute new coordinates and new minDistPhase1 ++++++++++
        // if minDistPhase1 =0, the H subgroup is reached
        g_phase1_nodes++;
        mv = 3 * search->ax[n] + search->po[n] - 1;
        search->flip[n + 1] = flipMove[search->flip[n]][mv];
        search->twist[n + 1] = twistMove[search->twist[n]][mv];
        search->slice[n + 1] = FRtoBR_Move[search->slice[n] * 24][mv] / 24;
        search->minDistPhase1[n + 1] = MAX(
            getPruning(Slice_Flip_Prun, N_SLICE1 * search->flip[n + 1] + search->slice[n + 1]),
            getPruning(Slice_Twist_Prun, N_SLICE1 * search->twist[n + 1] + search->slice[n + 1])
        );
        // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        // System.out.format("%d %d\n", n, depthPhase1);
        if (search->minDistPhase1[n + 1] == 0 && n >= depthPhase1 - 5) {
            search->minDistPhase1[n + 1] = 10;// instead of 10 any value >5 is possible
            if (n == depthPhase1 - 1 && (s = totalDepth(search, depthPhase1, maxDepth)) >= 0) {
                if (s == depthPhase1
                        || (search->ax[depthPhase1 - 1] != search->ax[depthPhase1] && search->ax[depthPhase1 - 1] != search->ax[depthPhase1] + 3)) {
                    char* res;
                    free((void*) fc);
                    free((void*) cc);
                    free((void*) c);
                    if (useSeparator) {
                        res = solutionToString(search, s, depthPhase1);
                    } else {
                        res = solutionToString(search, s, -1);
                    }
                    free((void*) search);
                    return res;
                }
            }

        }
    } while (1);
}

int totalDepth(search_t* search, int depthPhase1, int maxDepth)
{
    int mv = 0, d1 = 0, d2 = 0, i;
    int maxDepthPhase2 = MIN(10, maxDepth - depthPhase1);// Allow only max 10 moves in phase2
    int depthPhase2;
    int n;
    int busy;
    for (i = 0; i < depthPhase1; i++) {
        mv = 3 * search->ax[i] + search->po[i] - 1;
        // System.out.format("%d %d %d %d\n", i, mv, ax[i], po[i]);
        search->URFtoDLF[i + 1] = URFtoDLF_Move[search->URFtoDLF[i]][mv];
        search->FRtoBR[i + 1] = FRtoBR_Move[search->FRtoBR[i]][mv];
        search->parity[i + 1] = parityMove[search->parity[i]][mv];
    }

    if ((d1 = getPruning(Slice_URFtoDLF_Parity_Prun,
            (N_SLICE2 * search->URFtoDLF[depthPhase1] + search->FRtoBR[depthPhase1]) * 2 + search->parity[depthPhase1])) > maxDepthPhase2)
        return -1;

    for (i = 0; i < depthPhase1; i++) {
        mv = 3 * search->ax[i] + search->po[i] - 1;
        search->URtoUL[i + 1] = URtoUL_Move[search->URtoUL[i]][mv];
        search->UBtoDF[i + 1] = UBtoDF_Move[search->UBtoDF[i]][mv];
    }
    search->URtoDF[depthPhase1] = MergeURtoULandUBtoDF[search->URtoUL[depthPhase1]][search->UBtoDF[depthPhase1]];

    if ((d2 = getPruning(Slice_URtoDF_Parity_Prun,
            (N_SLICE2 * search->URtoDF[depthPhase1] + search->FRtoBR[depthPhase1]) * 2 + search->parity[depthPhase1])) > maxDepthPhase2)
        return -1;

    if ((search->minDistPhase2[depthPhase1] = MAX(d1, d2)) == 0)// already solved
        return depthPhase1;

    // now set up search

    depthPhase2 = 1;
    n = depthPhase1;
    busy = 0;
    search->po[depthPhase1] = 0;
    search->ax[depthPhase1] = 0;
    search->minDistPhase2[n + 1] = 1;// else failure for depthPhase2=1, n=0
    // +++++++++++++++++++ end initialization +++++++++++++++++++++++++++++++++
    do {
        do {
            if ((depthPhase1 + depthPhase2 - n > search->minDistPhase2[n + 1]) && !busy) {

                if (search->ax[n] == 0 || search->ax[n] == 3)// Initialize next move
                {
                    search->ax[++n] = 1;
                    search->po[n] = 2;
                } else {
                    search->ax[++n] = 0;
                    search->po[n] = 1;
                }
            } else if ((search->ax[n] == 0 || search->ax[n] == 3) ? (++search->po[n] > 3) : ((search->po[n] = search->po[n] + 2) > 3)) {
                do {// increment axis
                    if (++search->ax[n] > 5) {
                        if (n == depthPhase1) {
                            if (depthPhase2 >= maxDepthPhase2)
                                return -1;
                            else {
                                depthPhase2++;
                                search->ax[n] = 0;
                                search->po[n] = 1;
                                busy = 0;
                                break;
                            }
                        } else {
                            n--;
                            busy = 1;
                            break;
                        }

                    } else {
                        if (search->ax[n] == 0 || search->ax[n] == 3)
                            search->po[n] = 1;
                        else
                            search->po[n] = 2;
                        busy = 0;
                    }
                } while (n != depthPhase1 && (search->ax[n - 1] == search->ax[n] || search->ax[n - 1] - 3 == search->ax[n]));
            } else
                busy = 0;
        } while (busy);
        // +++++++++++++ compute new coordinates and new minDist ++++++++++
        mv = 3 * search->ax[n] + search->po[n] - 1;

        search->URFtoDLF[n + 1] = URFtoDLF_Move[search->URFtoDLF[n]][mv];
        search->FRtoBR[n + 1] = FRtoBR_Move[search->FRtoBR[n]][mv];
        search->parity[n + 1] = parityMove[search->parity[n]][mv];
        search->URtoDF[n + 1] = URtoDF_Move[search->URtoDF[n]][mv];

        search->minDistPhase2[n + 1] = MAX(getPruning(Slice_URtoDF_Parity_Prun, (N_SLICE2
                * search->URtoDF[n + 1] + search->FRtoBR[n + 1])
                * 2 + search->parity[n + 1]), getPruning(Slice_URFtoDLF_Parity_Prun, (N_SLICE2
                * search->URFtoDLF[n + 1] + search->FRtoBR[n + 1])
                * 2 + search->parity[n + 1]));
        // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    } while (search->minDistPhase2[n + 1] != 0);
    return depthPhase1 + depthPhase2;
}

// ========================================================================
// h_sort (oracle move ordering) Phase 1 — recursive IDA* variant.
// At every node we expand all 18 move candidates, compute each child's
// pruning-table h (admissible Phase 1 heuristic), and visit them in
// ascending-h order. This lets the native backend test whether the
// ordering ceiling (measured as 1.66x in Python) still pays off once the
// per-lookup cost drops from microseconds to nanoseconds.
// ========================================================================

typedef struct {
    int h;
    int ax;
    int po;
    int nf;
    int nt;
    int ns;
} hsort_cand_t;

static int hsort_phase1(search_t* search, int n, int depthPhase1, int maxDepth,
                        time_t tStart, long timeOut, int* out_sol)
{
    if (time(NULL) - tStart > timeOut) return -1;

    int parent_ax = (n > 0) ? search->ax[n - 1] : -99;
    hsort_cand_t scored[18];
    int nc = 0;

    // Enumerate all Phase-1 children after the parent-axis filter. Every child
    // whose coordinates we compute is counted as a Phase-1 node expansion, to
    // match the baseline solution()'s g_phase1_nodes++ accounting.
    for (int ax = 0; ax < 6; ax++) {
        if (n > 0 && (parent_ax == ax || parent_ax - 3 == ax)) continue;
        for (int po = 1; po <= 3; po++) {
            int mv = 3 * ax + (po - 1);
            int nf = flipMove[search->flip[n]][mv];
            int nt = twistMove[search->twist[n]][mv];
            int ns = FRtoBR_Move[search->slice[n] * 24][mv] / 24;
            int h = MAX(
                getPruning(Slice_Flip_Prun, N_SLICE1 * nf + ns),
                getPruning(Slice_Twist_Prun, N_SLICE1 * nt + ns)
            );
            g_phase1_nodes++; // count every expanded child, like the baseline
            scored[nc].h = h;
            scored[nc].ax = ax;
            scored[nc].po = po;
            scored[nc].nf = nf;
            scored[nc].nt = nt;
            scored[nc].ns = ns;
            nc++;
        }
    }

    // insertion sort by h ascending (nc <= 15)
    for (int i = 1; i < nc; i++) {
        hsort_cand_t x = scored[i];
        int j = i;
        while (j > 0 && scored[j - 1].h > x.h) {
            scored[j] = scored[j - 1];
            j--;
        }
        scored[j] = x;
    }

    for (int i = 0; i < nc; i++) {
        hsort_cand_t c = scored[i];
        int budget = depthPhase1 - (n + 1);
        if (c.h > budget) continue; // admissibility prune (already counted above)

        search->ax[n] = c.ax;
        search->po[n] = c.po;
        search->flip[n + 1] = c.nf;
        search->twist[n + 1] = c.nt;
        search->slice[n + 1] = c.ns;
        search->minDistPhase1[n + 1] = c.h;

        if (n + 1 == depthPhase1) {
            if (c.h != 0) continue;
            int s = totalDepth(search, depthPhase1, maxDepth);
            if (s >= 0) {
                if (s == depthPhase1 ||
                    (search->ax[depthPhase1 - 1] != search->ax[depthPhase1] &&
                     search->ax[depthPhase1 - 1] != search->ax[depthPhase1] + 3)) {
                    *out_sol = s;
                    return 1;
                }
            }
            continue;
        }

        int rc = hsort_phase1(search, n + 1, depthPhase1, maxDepth, tStart, timeOut, out_sol);
        if (rc == 1) return 1;
        if (rc == -1) return -1;
    }

    return 0;
}

char* solution_hsort(char* facelets, int maxDepth, long timeOut, int useSeparator, const char* cache_dir)
{
    search_t* search = (search_t*) calloc(1, sizeof(search_t));
    facecube_t* fc;
    cubiecube_t* cc;
    coordcube_t* c;

    int s, i;
    int count[6] = {0};
    time_t tStart;

    if (PRUNING_INITED == 0) {
        initPruning(cache_dir);
    }

    for (i = 0; i < 54; i++)
        switch (facelets[i]) {
            case 'U': count[U]++; break;
            case 'R': count[R]++; break;
            case 'F': count[F]++; break;
            case 'D': count[D]++; break;
            case 'L': count[L]++; break;
            case 'B': count[B]++; break;
        }
    for (i = 0; i < 6; i++)
        if (count[i] != 9) { free(search); return NULL; }

    fc = get_facecube_fromstring(facelets);
    cc = toCubieCube(fc);
    if ((s = verify(cc)) != 0) { free(search); return NULL; }

    c = get_coordcube(cc);
    search->flip[0] = c->flip;
    search->twist[0] = c->twist;
    search->parity[0] = c->parity;
    search->slice[0] = c->FRtoBR / 24;
    search->URFtoDLF[0] = c->URFtoDLF;
    search->FRtoBR[0] = c->FRtoBR;
    search->URtoUL[0] = c->URtoUL;
    search->UBtoDF[0] = c->UBtoDF;
    search->minDistPhase1[0] = MAX(
        getPruning(Slice_Flip_Prun, N_SLICE1 * search->flip[0] + search->slice[0]),
        getPruning(Slice_Twist_Prun, N_SLICE1 * search->twist[0] + search->slice[0])
    );

    g_phase1_nodes = 0;
    tStart = time(NULL);

    int depthPhase1;
    int sol_len = -1;
    int rc = 0;
    // Start iterative deepening at 1 to match baseline solution(). The
    // root-h seeded start would be a free bonus orthogonal to move ordering.
    for (depthPhase1 = 1; depthPhase1 <= maxDepth; depthPhase1++) {
        rc = hsort_phase1(search, 0, depthPhase1, maxDepth, tStart, timeOut, &sol_len);
        if (rc == 1) {
            char* res;
            free((void*) fc);
            free((void*) cc);
            free((void*) c);
            if (useSeparator) {
                res = solutionToString(search, sol_len, depthPhase1);
            } else {
                res = solutionToString(search, sol_len, -1);
            }
            free((void*) search);
            return res;
        }
        if (rc == -1) {
            free((void*) fc);
            free((void*) cc);
            free((void*) c);
            free((void*) search);
            return NULL; // timeout
        }
    }
    free((void*) fc);
    free((void*) cc);
    free((void*) c);
    free((void*) search);
    return NULL;
}

void patternize(char* facelets, char* pattern, char* patternized)
{
    facecube_t* fc;
    facecube_t* start_fc = get_facecube_fromstring(facelets);
    facecube_t* pattern_fc = get_facecube_fromstring(pattern);
    cubiecube_t* start_cc = toCubieCube(start_fc);
    cubiecube_t* pattern_cc = toCubieCube(pattern_fc);
    cubiecube_t* inv_pattern_cc = get_cubiecube();
    invCubieCube(pattern_cc, inv_pattern_cc);
    multiply(inv_pattern_cc, start_cc);
    fc = toFaceCube(inv_pattern_cc);
    to_String(fc, patternized);
    free(start_fc);
    free(pattern_fc);
    free(start_cc);
    free(pattern_cc);
    free(inv_pattern_cc);
    free(fc);
}
