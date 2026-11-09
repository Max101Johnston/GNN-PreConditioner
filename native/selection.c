/* Directed strong-path exclusion on an incoming CSR, in priority order. */
#include <stdint.h>
#include <stdlib.h>

int gnn_select(int n, int count, const int32_t *order, const uint8_t *eligible,
               const int32_t *offsets, const int32_t *rows, int radius,
               int8_t *distance, uint8_t *selected)
{
    int32_t *queue = malloc((n ? n : 1) * sizeof(*queue));
    if (!queue) return 1;
    for (int k = 0; k < count; ++k)
    {
        int root = order[k];
        if (distance[root] <= radius) continue;
        selected[root] = 1;
        distance[root] = 0;
        int tail = 1;
        queue[0] = root;
        for (int head = 0; head < tail; ++head)
        {
            int node = queue[head];
            int next = distance[node] + 1;
            if (next > radius) continue;
            for (int j = offsets[node]; j < offsets[node + 1]; ++j)
            {
                int row = rows[j];
                if (!eligible[row] || distance[row] <= next) continue;
                distance[row] = next;
                queue[tail++] = row;
            }
        }
    }
    free(queue);
    return 0;
}
