/* External C/F interpolation bridge for PETSc 3.16.3 / HYPRE 2.23.0.
 * No BoomerAMG setup or coarsening routine is called here.
 */
#include <petscmathypre.h>
#include <_hypre_parcsr_ls.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

_Static_assert(PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR == 16 &&
                   PETSC_VERSION_SUBMINOR == 3, "PETSc ABI mismatch");
_Static_assert(HYPRE_RELEASE_NUMBER == 22300, "HYPRE ABI mismatch");
_Static_assert(sizeof(PetscInt) == 4 && sizeof(PetscScalar) == 8 &&
                   sizeof(HYPRE_Int) == 4, "scalar/index ABI mismatch");

/* Avoid PETSc 3.16 MatCreateFromParCSR's empty-column ownership adjustment.
 * C-point counts can be zero on individual MPI ranks. */
static PetscErrorCode import_parcsr(hypre_ParCSRMatrix *h, Mat *a)
{
    hypre_CSRMatrix *d = hypre_ParCSRMatrixDiag(h);
    hypre_CSRMatrix *o = hypre_ParCSRMatrixOffd(h);
    PetscInt m = hypre_CSRMatrixNumRows(d), n = hypre_CSRMatrixNumCols(d);
    PetscInt *dn = NULL, *on = NULL, *columns = NULL;
    PetscScalar *values = NULL;
    PetscInt capacity = 0;
    PetscErrorCode ierr;
    ierr = PetscMalloc2(m, &dn, m, &on);CHKERRQ(ierr);
    for (PetscInt i = 0; i < m; ++i)
    {
        dn[i] = hypre_CSRMatrixI(d)[i+1] - hypre_CSRMatrixI(d)[i];
        on[i] = hypre_CSRMatrixI(o)[i+1] - hypre_CSRMatrixI(o)[i];
        capacity = PetscMax(capacity, dn[i] + on[i]);
    }
    ierr = MatCreateAIJ(hypre_ParCSRMatrixComm(h), m, n,
        hypre_ParCSRMatrixGlobalNumRows(h), hypre_ParCSRMatrixGlobalNumCols(h),
        0, dn, 0, on, a);CHKERRQ(ierr);
    ierr = PetscMalloc2(capacity, &columns, capacity, &values);CHKERRQ(ierr);
    for (PetscInt i = 0; i < m; ++i)
    {
        PetscInt row = hypre_ParCSRMatrixFirstRowIndex(h) + i, k = 0;
        for (PetscInt j = hypre_CSRMatrixI(d)[i]; j < hypre_CSRMatrixI(d)[i+1]; ++j)
        {
            columns[k] = hypre_ParCSRMatrixFirstColDiag(h) + hypre_CSRMatrixJ(d)[j];
            values[k++] = hypre_CSRMatrixData(d)[j];
        }
        for (PetscInt j = hypre_CSRMatrixI(o)[i]; j < hypre_CSRMatrixI(o)[i+1]; ++j)
        {
            columns[k] = hypre_ParCSRMatrixColMapOffd(h)[hypre_CSRMatrixJ(o)[j]];
            values[k++] = hypre_CSRMatrixData(o)[j];
        }
        ierr = MatSetValues(*a, 1, &row, k, columns, values, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(*a, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*a, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscFree2(columns, values);CHKERRQ(ierr);
    ierr = PetscFree2(dn, on);CHKERRQ(ierr);
    return 0;
}

/* Propagate synchronously along actual directed strong dependencies. */
static long long propagate(hypre_ParCSRMatrix *s, int *local, int *global,
                           const int *cf, const int *counts, const int *offsets,
                           int passes)
{
    MPI_Comm comm = hypre_ParCSRMatrixComm(s);
    hypre_CSRMatrix *d = hypre_ParCSRMatrixDiag(s), *o = hypre_ParCSRMatrixOffd(s);
    int m = hypre_CSRMatrixNumRows(d), first = hypre_ParCSRMatrixFirstRowIndex(s);
    for (int pass = 0; pass < passes; ++pass)
    {
        (void)MPI_Allgatherv(local, m, MPI_INT, global, counts, offsets, MPI_INT, comm);
        int changed = 0, total_changed;
        for (int i = 0; i < m; ++i)
        {
            if (local[i] || cf[i] == -3) continue;
            for (int j = hypre_CSRMatrixI(d)[i]; j < hypre_CSRMatrixI(d)[i+1]; ++j)
            {
                int c = hypre_CSRMatrixJ(d)[j];
                if (c >= 0 && global[first+c]) local[i] = 1;
            }
            for (int j = hypre_CSRMatrixI(o)[i]; j < hypre_CSRMatrixI(o)[i+1]; ++j)
            {
                int c = hypre_CSRMatrixJ(o)[j];
                if (c >= 0 && global[hypre_ParCSRMatrixColMapOffd(s)[c]]) local[i] = 1;
            }
            changed += local[i];
        }
        MPI_Allreduce(&changed, &total_changed, 1, MPI_INT, MPI_SUM, comm);
        if (!total_changed) break;
    }
    (void)MPI_Allgatherv(local, m, MPI_INT, global, counts, offsets, MPI_INT, comm);
    long long here = 0, missing;
    for (int i = 0; i < m; ++i) here += cf[i] == -1 && !local[i];
    MPI_Allreduce(&here, &missing, 1, MPI_LONG_LONG, MPI_SUM, comm);
    return missing;
}

/* Fixed environment repair. Preserve every proposed C point, then add only
 * uncovered F points. Native IDs break ties independently of MPI partitioning.
 * Simultaneous local minima among uncovered outgoing neighbours are promoted;
 * repeat only if the actual nine-hop audit still fails. This is never called
 * as a replacement coarsener: the caller requires a nonempty proposal and caps
 * additions. Raw 9/99-hop diagnostics and all promoted rows remain observable.
 * flags: 1 originally uncovered at 9 hops, 2 still uncovered at 99 hops,
 *        4 promoted by repair. An exhausted budget is a failed candidate. */
static int uncovered(hypre_ParCSRMatrix *s, int *cf, const int64_t *native,
                     int max_repairs, int *flags, long long stats[11])
{
    MPI_Comm comm = hypre_ParCSRMatrixComm(s);
    hypre_CSRMatrix *d = hypre_ParCSRMatrixDiag(s), *o = hypre_ParCSRMatrixOffd(s);
    int ranks, m = hypre_CSRMatrixNumRows(d), n = hypre_ParCSRMatrixGlobalNumRows(s);
    int first = hypre_ParCSRMatrixFirstRowIndex(s);
    MPI_Comm_size(comm, &ranks);
    int *counts = calloc(ranks, sizeof(int)), *offsets = calloc(ranks, sizeof(int));
    int *local = calloc(m ? m : 1, sizeof(int)), *global = calloc(n ? n : 1, sizeof(int));
    int *missing = calloc(n ? n : 1, sizeof(int)), *candidate = calloc(m ? m : 1, sizeof(int));
    int64_t *ids = calloc(n ? n : 1, sizeof(int64_t));
    int ok = counts && offsets && local && global && missing && candidate && ids, all_ok;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
    if (!all_ok) { free(counts); free(offsets); free(local); free(global);
        free(missing); free(candidate); free(ids); return 55; }
    (void)MPI_Allgather(&m, 1, MPI_INT, counts, 1, MPI_INT, comm);
    for (int i = 1; i < ranks; ++i) offsets[i] = offsets[i-1] + counts[i-1];
    (void)MPI_Allgatherv(native, m, MPI_INT64_T, ids, counts, offsets, MPI_INT64_T, comm);
    for (int i = 0; i < m; ++i) { local[i] = cf[i] == 1; flags[i] = 0; }
    stats[6] = propagate(s, local, global, cf, counts, offsets, 9);
    for (int i = 0; i < m; ++i) if (cf[i] == -1 && !local[i]) flags[i] |= 1;
    stats[7] = stats[6] ? propagate(s, local, global, cf, counts, offsets, 90) : 0;
    for (int i = 0; i < m; ++i) if (cf[i] == -1 && !local[i]) flags[i] |= 2;
    stats[2] = stats[6];
    while (stats[2])
    {
        for (int i = 0; i < m; ++i) local[i] = cf[i] == 1;
        stats[2] = propagate(s, local, global, cf, counts, offsets, 9);
        if (!stats[2]) break;
        for (int i = 0; i < m; ++i) candidate[i] = cf[i] == -1 && !local[i];
        (void)MPI_Allgatherv(candidate, m, MPI_INT, missing, counts, offsets, MPI_INT, comm);
        long long here = 0, total;
        for (int i = 0; i < m; ++i)
        {
            if (!candidate[i]) continue;
            for (int j = hypre_CSRMatrixI(d)[i]; j < hypre_CSRMatrixI(d)[i+1]; ++j)
            {
                int c = hypre_CSRMatrixJ(d)[j];
                if (c >= 0 && missing[first+c] && ids[first+c] < native[i]) candidate[i] = 0;
            }
            for (int j = hypre_CSRMatrixI(o)[i]; j < hypre_CSRMatrixI(o)[i+1]; ++j)
            {
                int c = hypre_CSRMatrixJ(o)[j];
                if (c >= 0) { int g = hypre_ParCSRMatrixColMapOffd(s)[c];
                    if (missing[g] && ids[g] < native[i]) candidate[i] = 0; }
            }
            here += candidate[i];
        }
        MPI_Allreduce(&here, &total, 1, MPI_LONG_LONG, MPI_SUM, comm);
        if (!total || stats[8]+total > max_repairs || stats[9] >= 16)
        { stats[10] = 1; break; }
        for (int i = 0; i < m; ++i) if (candidate[i]) { cf[i] = 1; flags[i] |= 4; }
        stats[8] += total;
        ++stats[9];
    }
    free(counts); free(offsets); free(local); free(global);
    free(missing); free(candidate); free(ids);
    return 0;
}

/* stats: input rows, coarse columns, uncovered F rows, empty F rows, P nnz,
 * newly classified special F rows, raw uncovered at 9/99 hops,
 * added C points, repair rounds, exhausted-repair-budget flag.
 * The caller supplies actual component IDs; coarse IDs are NOT index modulo 3.
 * Constrained rows use -3 and are handled by the unchanged fine smoother. */
PetscErrorCode gnn_multipass(Mat a, int *markers, const int *functions,
                            double strength, double truncation, int max_entries,
                            const int64_t *native, int max_repairs, int *flags,
                            Mat *p, Mat *coarse, long long stats[11])
{
    Mat hmat = NULL;
    hypre_ParCSRMatrix *h = NULL, *s = NULL, *hp = NULL;
    hypre_IntArray *cf = NULL, *dof = NULL, *cdof = NULL;
    HYPRE_BigInt *starts = NULL;
    PetscErrorCode err = 0;
    PetscInt m, lo, hi;
    *p = NULL; *coarse = NULL;
    for (int i = 0; i < 11; ++i) stats[i] = 0;
#define TRY(call) do { err = (call); if (err) goto cleanup; } while (0)
    TRY(MatConvert(a, MATHYPRE, MAT_INITIAL_MATRIX, &hmat));
    TRY(MatHYPREGetParCSR(hmat, &h));
    m = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(h));
    cf = hypre_IntArrayCreate(m); dof = hypre_IntArrayCreate(m);
    TRY(hypre_IntArrayInitialize(cf)); TRY(hypre_IntArrayInitialize(dof));
    for (PetscInt i = 0; i < m; ++i)
    {
        hypre_IntArrayData(cf)[i] = markers[i];
        hypre_IntArrayData(dof)[i] = functions[i];
    }
    TRY(hypre_BoomerAMGCreateS(h, strength, 0.9, 3, hypre_IntArrayData(dof), &s));
    /* Match par_coarsen.c: no outgoing strong neighbours => SF_PT.
     * This is the existing numerical special-F convention, not a learned
     * deletion or an excuse to overlook an uncovered interpolatable row. */
    long long local_special = 0;
    hypre_CSRMatrix *sd = hypre_ParCSRMatrixDiag(s), *so = hypre_ParCSRMatrixOffd(s);
    for (PetscInt i = 0; i < m; ++i)
    {
        PetscInt degree = hypre_CSRMatrixI(sd)[i+1] - hypre_CSRMatrixI(sd)[i]
                        + hypre_CSRMatrixI(so)[i+1] - hypre_CSRMatrixI(so)[i];
        if (markers[i] == -1 && degree == 0)
        {
            markers[i] = -3;
            hypre_IntArrayData(cf)[i] = -3;
            ++local_special;
        }
    }
    TRY(MPI_Allreduce(&local_special, &stats[5], 1, MPI_LONG_LONG, MPI_SUM, hypre_ParCSRMatrixComm(h)));
    TRY(uncovered(s, markers, native, max_repairs, flags, stats));
    for (PetscInt i = 0; i < m; ++i) hypre_IntArrayData(cf)[i] = markers[i];
    TRY(hypre_BoomerAMGCoarseParms(hypre_ParCSRMatrixComm(h), m, 3, dof, cf, &cdof, &starts));
    TRY(hypre_BoomerAMGBuildMultipass(h, hypre_IntArrayData(cf), s, starts, 3,
        hypre_IntArrayData(dof), 0, truncation, max_entries, 0, &hp));
    TRY(import_parcsr(hp, p));
    stats[0] = hypre_ParCSRMatrixGlobalNumRows(hp);
    stats[1] = hypre_ParCSRMatrixGlobalNumCols(hp);
    long long local_stats[3] = {0, 0, 0}, total[3];
    TRY(MatGetOwnershipRange(*p, &lo, &hi));
    for (PetscInt row = lo; row < hi; ++row)
    {
        PetscInt count; const PetscInt *columns; const PetscScalar *values;
        TRY(MatGetRow(*p, row, &count, &columns, &values));
        local_stats[0] += count == 0 && markers[row-lo] == -1;
        local_stats[1] += count;
        for (PetscInt j = 0; j < count; ++j) if (!isfinite(values[j])) local_stats[2]++;
        TRY(MatRestoreRow(*p, row, &count, &columns, &values));
    }
    TRY(MPI_Allreduce(local_stats, total, 3, MPI_LONG_LONG, MPI_SUM, hypre_ParCSRMatrixComm(h)));
    stats[3] = total[0]; stats[4] = total[1];
    if (total[2]) { err = 72; goto cleanup; }
    TRY(MatPtAP(a, *p, MAT_INITIAL_MATRIX, PETSC_DEFAULT, coarse));
cleanup:
    if (hp) hypre_ParCSRMatrixDestroy(hp);
    if (s) hypre_ParCSRMatrixDestroy(s);
    if (cf) hypre_IntArrayDestroy(cf);
    if (dof) hypre_IntArrayDestroy(dof);
    if (cdof) hypre_IntArrayDestroy(cdof);
    if (starts) hypre_TFree(starts, HYPRE_MEMORY_HOST);
    if (hmat) MatDestroy(&hmat);
    if (err) { if (*p) MatDestroy(p); if (*coarse) MatDestroy(coarse); }
    return err;
#undef TRY
}
