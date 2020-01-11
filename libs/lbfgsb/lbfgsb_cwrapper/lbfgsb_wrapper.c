#include "lbfgsb_wrapper.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

extern void setulb_(int *n, int *m, double x[], double l[],
                    double u[], int nbd[], double *f, double g[],
                    double *factr, double *pgtol, double wa[],
                    int iwa[], char task[], int *iprint,
                    char csave[], int lsave[], int isave[],
                    double dsave[]);

struct lbfgsb *create(int n, int m) {
    struct lbfgsb *data = malloc(sizeof(struct lbfgsb));
    data->n = n;
    data->m = m;
    data->x = malloc(n * sizeof(double));
    data->l = malloc(n * sizeof(double));
    data->u = malloc(n * sizeof(double));
    data->nbd = malloc(n * sizeof(int));
    data->f = 0;
    data->g = malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) data->g[i] = 0;
    data->factr = 1e7;
    data->pgtol = 1e-5;
    int wa_size = 2 * m * n + 5 * n + 11 * m * m + 8 * m;
    data->wa = malloc(wa_size * sizeof(double));
    data->iwa = malloc(3 * n * sizeof(int));
    //hack to make a proper C-string from Fortran's space-padded string
    data->task[LBFGSB_TASK_SIZE] = '\0';
    data->iprint = -1;
    return data;
}

void close(struct lbfgsb *data) {
    free(data->x);
    free(data->l);
    free(data->u);
    free(data->nbd);
    free(data->g);
    free(data->wa);
    free(data->iwa);
    free(data);
}

void step(struct lbfgsb *data) {
    setulb_(&(data->n), &(data->m), data->x, data->l, data->u, data->nbd,
            &(data->f), data->g, &(data->factr), &(data->pgtol),
            data->wa, data->iwa, data->task, &(data->iprint),
            data->csave, data->lsave, data->isave, data->dsave);
}

void set_task(struct lbfgsb *data, enum TaskType type) {
    switch (type) {
        case LBFGSB_FG:
            set_task_str(data, "FG");
            break;
        case LBFGSB_NEW_X:
            set_task_str(data, "NEW_X");
            break;
        case LBFGSB_CONV:
            set_task_str(data, "CONV");
            break;
        case LBFGSB_ABNO:
            set_task_str(data, "ABNO");
            break;
        case LBFGSB_ERROR:
            set_task_str(data, "ERROR");
            break;
        case LBFGSB_START:
            set_task_str(data, "START");
            break;
        case LBFGSB_STOP:
            set_task_str(data, "STOP");
            break;
        default: {
            printf("Exception: set_task: Unknown task type\n");
            exit(1);
        }
    }
}

enum TaskType get_task(struct lbfgsb *data) {
    if (is_task_str_equal(data, "FG")) return LBFGSB_FG;
    if (is_task_str_equal(data, "NEW_X")) return LBFGSB_NEW_X;
    if (is_task_str_equal(data, "CONV")) return LBFGSB_CONV;
    if (is_task_str_equal(data, "ABNO")) return LBFGSB_ABNO;
    if (is_task_str_equal(data, "ERROR")) return LBFGSB_ERROR;
    if (is_task_str_equal(data, "START")) return LBFGSB_START;
    if (is_task_str_equal(data, "STOP")) return LBFGSB_STOP;
    else return LBFGSB_UNKNOWN;
}

void set_task_str(struct lbfgsb *data, char *c_str) {
    int len = strlen(c_str);
    assert(len < LBFGSB_TASK_SIZE && "Exception: is_task_str_equal: String too long\n");
    memset(data->task, ' ', LBFGSB_TASK_SIZE);
    memcpy(data->task, c_str, len);
}

int is_task_str_equal(struct lbfgsb *data, char *c_str) {
    int len = strlen(c_str);
    assert(len < LBFGSB_TASK_SIZE && "Exception: is_task_str_equal: String too long\n");
    return memcmp(data->task, c_str, len) == 0;
}
