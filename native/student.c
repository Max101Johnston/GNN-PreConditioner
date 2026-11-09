/* Small signed interpolation students. All arrays contain owned rows only. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

enum { ROW_FEATURES = 10, EDGE_FEATURES = 7, MAX_WIDTH = 16, MAX_CANDIDATES = 32 };

struct candidate {
    int32_t col;
    double base, step1, step2, priority;
};

static int priority_order(const void *x, const void *y)
{
    const struct candidate *a = x, *b = y;
    if (a->priority > b->priority) return -1;
    if (a->priority < b->priority) return 1;
    return (a->col > b->col) - (a->col < b->col);
}

/* Merge signed values of P0, K P0, and the optional bounded second path.
 * The first two matrices retain full PETSc cross-rank multiplication. */
int gnn_pack_candidates(int n, int cap, const int32_t *markers, const double *rowsum,
    const int32_t *p_i, const int32_t *p_j, const double *p_v,
    const int32_t *a_i, const int32_t *a_j, const double *a_v,
    const int32_t *b_i, const int32_t *b_j, const double *b_v,
    float *row_features, int32_t *columns, float *hints, float *norm)
{
    if (cap < 12 || cap > MAX_CANDIDATES) return 1;
    int maxlen = 1;
    for (int i = 0; i < n; ++i)
    {
        int len = (p_i[i+1]-p_i[i]) + (a_i[i+1]-a_i[i]) + (b_i[i+1]-b_i[i]);
        if (len > maxlen) maxlen = len;
    }
    struct candidate *row = malloc((size_t)maxlen * sizeof(*row));
    if (!row) return 2;
    for (int i = 0; i < n; ++i)
    {
        int len = 0, j = p_i[i], a = a_i[i], b = b_i[i];
        double scale = 1.0/fmax(rowsum[i], 1e-30), maxbase = 0.0, base_sum = 0.0;
        int ae = markers[i] == -1 ? a_i[i+1] : a, be = markers[i] == -1 ? b_i[i+1] : b;
        for (int k = j; k < p_i[i+1]; ++k)
        {
            maxbase = fmax(maxbase, fabs(p_v[k]));
            base_sum += fabs(p_v[k]);
        }
        while (j < p_i[i+1] || a < ae || b < be)
        {
            int col = INT32_MAX;
            if (j < p_i[i+1] && p_j[j] < col) col = p_j[j];
            if (a < ae && a_j[a] < col) col = a_j[a];
            if (b < be && b_j[b] < col) col = b_j[b];
            double x = (j < p_i[i+1] && p_j[j] == col) ? p_v[j++] : 0.0;
            double y = (a < ae && a_j[a] == col) ? -scale*a_v[a++] : 0.0;
            double z = (b < be && b_j[b] == col) ? -b_v[b++] : 0.0;
            double score = fmax(fabs(x), fmax(fabs(x+y), fabs(x+y+z)));
            if (score > 0.0)
                row[len++] = (struct candidate){col, x, y, z, score};
        }
        if (len > 1) qsort(row, (size_t)len, sizeof(*row), priority_order);
        int keep = len < cap ? len : cap;
        double row_norm = 1e-8;
        for (int k = 0; k < cap; ++k)
        {
            int offset = i*cap+k;
            columns[offset] = k < keep ? row[k].col : -1;
            hints[3*offset] = k < keep ? (float)row[k].base : 0.0f;
            hints[3*offset+1] = k < keep ? (float)row[k].step1 : 0.0f;
            hints[3*offset+2] = k < keep ? (float)row[k].step2 : 0.0f;
            if (k < keep) row_norm = fmax(row_norm, row[k].priority);
        }
        norm[i] = (float)row_norm;
        int n0 = p_i[i+1]-p_i[i];
        float *f = row_features + i*ROW_FEATURES;
        f[6] = (float)n0/8.0f;
        f[7] = (float)(base_sum/fmax(maxbase*fmax(n0, 1), 1e-20));
        f[8] = (float)log1p(maxbase);
        f[9] = (float)log1p(row_norm);
    }
    free(row);
    return 0;
}

static float clip(float x, float lo, float hi)
{
    return fminf(fmaxf(x, lo), hi);
}

/* Packed layers are W1, b1, W2, b2, in row-major output order. */
static void mlp(const float *x, int inputs, int width, int outputs, const float *params, float *y)
{
    const float *b1 = params + width*inputs;
    const float *w2 = b1 + width;
    const float *b2 = w2 + outputs*width;
    float hidden[MAX_WIDTH];
    for (int i = 0; i < width; ++i)
    {
        float t = b1[i];
        for (int j = 0; j < inputs; ++j) t += params[i*inputs+j]*x[j];
        hidden[i] = fmaxf(t, 0.0f);
    }
    for (int i = 0; i < outputs; ++i)
    {
        float t = b2[i];
        for (int j = 0; j < width; ++j) t += w2[i*width+j]*hidden[j];
        y[i] = t;
    }
}

int gnn_student_predict(int n, int cap, int width, int kind, const float *params,
    const float *control, const int32_t *markers, const int32_t *columns,
    const int32_t *coarse_component, const float *rows, const float *hints,
    const float *norm, float *values, uint8_t *budget)
{
    if (cap < 12 || cap > MAX_CANDIDATES || width < 1 || width > MAX_WIDTH ||
        kind < 0 || kind > 1) return 1;
    const float *edge = params + (ROW_FEATURES+5)*width+4;
    for (int i = 0; i < n; ++i)
    {
        float score[MAX_CANDIDATES], weight[MAX_CANDIDATES], r[4];
        const float *x = rows+i*ROW_FEATURES;
        budget[i] = 0;
        for (int j = 0; j < cap; ++j) values[i*cap+j] = 0.0f;
        if (markers[i] != -1) continue;
        mlp(x, ROW_FEATURES, width, 4, params, r);
        float omega = 1.0f + clip(r[0]*control[2], -0.5f, 0.5f);
        float logits[] = {r[1]+control[0], r[2]+control[1], r[3]};
        int choice = 0;
        for (int k = 1; k < 3; ++k) if (logits[k] > logits[choice]) choice = k;
        const uint8_t caps[] = {6, 8, 12};
        budget[i] = caps[choice];
        float scale = norm[i];
        for (int j = 0; j < cap; ++j)
        {
            int k = i*cap+j, col = columns[k];
            score[j] = -INFINITY;
            weight[j] = 0.0f;
            if (col < 0) continue;
            const float *a = hints + 3*k;
            float value = a[0] + omega*a[1];
            float shift = 0.0f;
            if (kind == 1)
            {
                float z[] = {a[0]/scale, a[1]/scale, a[2]/scale,
                    2.0f*x[5] == coarse_component[col] ? 1.0f : 0.0f,
                    x[1], x[4], x[6]};
                float e[2];
                mlp(z, EDGE_FEATURES, width, 2, edge, e);
                value += a[2];
                value += scale*clip(e[0], -1.0f, 1.0f)*control[3];
                shift = clip(e[1], -1.0f, 1.0f)*control[3];
            }
            if (!isfinite(value) || !isfinite(shift)) return 2;
            weight[j] = value;
            if (value != 0.0f) score[j] = fabsf(value)/scale + shift;
        }
        for (int k = 0; k < budget[i]; ++k)
        {
            int best = -1;
            float top = -INFINITY;
            for (int j = 0; j < cap; ++j)
                if (score[j] > top) { top = score[j]; best = j; }
            if (best < 0) break;
            values[i*cap+best] = weight[best];
            score[best] = -INFINITY;
        }
    }
    return 0;
}
