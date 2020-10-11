#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

double dx = 0.000000001;
int NUM_THREADS = 4;

void OpemMP_Private() {

    int n = 1 / dx, i;
    double pi = 0.0, sum[NUM_THREADS];

    omp_set_num_threads(NUM_THREADS);

    double start = omp_get_wtime();

    #pragma omp parallel private(i)
    {
        int id = omp_get_thread_num();
        sum[id] = 0;

        for (i = id; i < n; i += NUM_THREADS){
            
            sum[id] += dx * (1.0 / (1.0 + i * dx * dx * i));
        }
    }

    for (i = 0; i < NUM_THREADS; ++i) {

        pi += sum[i];
    }
    
    double time_used = omp_get_wtime() - start;

    printf("CPU parallel using private OPENMP method time used %lf ms\n", time_used * 1000);

    pi = 4 * pi;

    printf("pi = %.20lf \n\n", pi);
}

void Open_MP_shard_mission() {

    int n = 1 / dx, i;
    double pi = 0.0, sum[NUM_THREADS];

    omp_set_num_threads(NUM_THREADS);

    double start = omp_get_wtime();

    #pragma omp parallel 
    {
        int id = omp_get_thread_num();
        sum[id] = 0;

        #pragma omp for
        for (i = 0; i < n; ++i){
            
            sum[id] += dx * (1.0 / (1.0 + i * dx * dx * i));
        }
    }

    for (i = 0; i < NUM_THREADS; ++i) {

        pi += sum[i];
    }
    
    double time_used = omp_get_wtime() - start;

    printf("CPU parallel using shared OPENMP method time used %lf ms\n", time_used * 1000);

    pi = 4 * pi;

    printf("pi = %.20lf \n\n", pi);
}

void OpenMP_Reduction() {// using OpenMP reduction method

    double pi = 0.0;
    int n = 1 / dx, i;

    omp_set_num_threads(NUM_THREADS);

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:pi)
    for (i = 0; i < n; ++i) {

        pi += dx * (1.0 / (1.0 + i * dx * dx * i));
    }

    double time_used = omp_get_wtime() - start;

    printf("CPU parallel using reduction OPENMP method time used %lf ms\n", time_used * 1000);

    pi = 4 * pi;

    printf("pi = %.20lf \n\n", pi);
}

void normal() {// using normal method

    double pi = 0.0;
    int n = 1 / dx, i;

    clock_t start = clock();

    for (i = 0; i < n; ++i) {

        pi += dx * (1.0 / (1.0 + i * dx * dx * i));
    }

    clock_t time_used = clock() - start;

    printf("CPU using common method time used %lf ms\n", (double)(time_used) / CLOCKS_PER_SEC * 1000);\

    pi = 4 * pi;

    printf("pi = %.20lf \n\n", pi);
}

int main(int argc, char *argv[]) {

    OpemMP_Private();
    Open_MP_shard_mission();
    OpenMP_Reduction();
    normal();
}