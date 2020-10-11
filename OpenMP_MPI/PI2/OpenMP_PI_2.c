#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>

int NUM_THREADS = 4;

void OpemMP_Private() {

    double pi = 0.0, round = 0, x, y, dist, sum[NUM_THREADS];
    int i, n = 100000000;

    omp_set_num_threads(NUM_THREADS);

    double start = omp_get_wtime();

    #pragma omp parallel private(i)
    {
        int id = omp_get_thread_num();
        sum[id] = 0;

        for (i = id; i < n; i += NUM_THREADS) {

            x = 2.0 * rand() / RAND_MAX - 1.0;
            y = 2.0 * rand() / RAND_MAX - 1.0;
            
            dist = sqrt(x * x + y * y);

            if (dist <= 1) {

                sum[id] += 1;
            }
        }
    }

    for (i = 0; i < NUM_THREADS; ++i) {

        round += sum[i];
    }
    
    double time_used = omp_get_wtime() - start;

    printf("CPU parallel using private OPENMP method time used %lf ms\n", time_used * 1000);

    pi = 4 * round / n;

    printf("pi = %.20lf \n\n", pi);
}

void Open_MP_shard_mission() {

    double pi = 0.0, round = 0, x, y, dist, sum[NUM_THREADS];
    int i, n = 100000000;

    omp_set_num_threads(NUM_THREADS);

    double start = omp_get_wtime();

    #pragma omp parallel 
    {
        int id = omp_get_thread_num();
        sum[id] = 0;

        #pragma omp for
        for (i = 0; i < n; ++i) {

            x = 2.0 * rand() / RAND_MAX - 1.0;
            y = 2.0 * rand() / RAND_MAX - 1.0;
            
            dist = sqrt(x * x + y * y);

            if (dist <= 1) {

                sum[id] += 1;
            }
        }
    }

    for (i = 0; i < NUM_THREADS; ++i) {

        round += sum[i];
    }
    
    double time_used = omp_get_wtime() - start;

    printf("CPU parallel using shared OPENMP method time used %lf ms\n", time_used * 1000);

    pi = 4 * round / n;

    printf("pi = %.20lf \n\n", pi);
}

void OpenMP_Reduction() {// using OpenMP reduction method

    double pi = 0.0, round = 0, x, y, dist;
    int i, n = 100000000;

    omp_set_num_threads(NUM_THREADS);

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:round)
    for (i = 0; i < n; ++i) {

        x = 2.0 * rand() / RAND_MAX - 1.0;
        y = 2.0 * rand() / RAND_MAX - 1.0;
        
        dist = sqrt(x * x + y * y);

        if (dist <= 1) {

            round += 1;
        }
    }

    double time_used = omp_get_wtime() - start;

    printf("CPU parallel using reduction OPENMP method time used %lf ms\n", time_used * 1000);

    pi = 4 * round / n;

    printf("pi = %.20lf \n\n", pi);
}

void normal() {// using normal method

    double pi = 0.0, round = 0;
    int i, n = 100000000;

    srand((unsigned)time(NULL));

    clock_t start = clock();

    for (i = 0.0; i < n; ++i) {

        double x = 2.0 * rand() / RAND_MAX - 1.0;
        double y = 2.0 * rand() / RAND_MAX - 1.0;
        
        double dist = sqrt(x * x + y * y);

        if (dist <= 1) {

            round += 1;
        }
    }
    
    clock_t time_used = clock() - start;

    printf("CPU using common method time used %lf ms\n", (double)(time_used) / CLOCKS_PER_SEC * 1000);

    pi = 4 * round / n;

    printf("pi = %.20lf \n\n", pi);
}

int main(int argc, char *argv[]) {

    OpemMP_Private();
    Open_MP_shard_mission();
    OpenMP_Reduction();
    normal();
}