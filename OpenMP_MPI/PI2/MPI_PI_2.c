#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

double dx = 0.000000001;

void normal() {// using normal method

    double pi = 0.0, round = 0.0, x, y, dist;
    int i, n = 100000000;

    srand((unsigned)time(NULL));

    clock_t start = clock();

    for (i = 0; i < n; ++i) {

        x = 2.0 * rand() / RAND_MAX - 1.0;
        y = 2.0 * rand() / RAND_MAX - 1.0;
        
        dist = sqrt(x * x + y * y);

        if (dist <= 1) {

            round += 1;
        }
    }
    
    clock_t time_used = clock() - start;

    printf("CPU using common method time used %lf ms\n", (double)(time_used) / CLOCKS_PER_SEC * 1000);

    pi = 4 * round / n;

    printf("pi = %.20lf \n\n", pi);
}

void MPI(int argc, char *argv[]) {// using MPI method

    double pi = 0.0, round_per_rank = 0.0, start, time_used, x, y, dist;
    int n = 100000000, i, rank, size, round = 0;

    MPI_Status status;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {

        start = MPI_Wtime();
    }
    
    for (i = rank; i < n; i += size) {

        x = 2.0 * rand() / RAND_MAX - 1.0;
        y = 2.0 * rand() / RAND_MAX - 1.0;
        
        dist = sqrt(x * x + y * y);

        if (dist <= 1) {

            round_per_rank += 1;
        }
    }

    if (rank != 0) {

        MPI_Send(&round_per_rank, 1, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD); 
    }

    if(rank == 0) {

        round = round_per_rank;

        for (i = 1; i < size; ++i) {

            MPI_Recv(&round_per_rank, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);

            round += round_per_rank;
        }

        time_used = MPI_Wtime() - start;    
        
        printf("CPU using MPI method time used %lf ms\n", time_used * 1000);\

        pi = 4.0 * round / n;

        printf("pi = %.20lf \n\n", pi);
    }

    MPI_Finalize();

    if (rank == 0) {

        normal();
    }
}

int main(int argc, char *argv[]) {

    MPI(argc, argv);
}