#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

double dx = 0.000000001;

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

void MPI(int argc, char *argv[]) {// using MPI method

    double pi_per_rank = 0.0, pi = 0.0, start, time_used;
    int n = 1 / dx, i, rank, size;

    MPI_Status status;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {

        start = MPI_Wtime();
    }
    
    for (i = rank; i < n; i += size) {

        pi_per_rank += dx * (1.0 / (1.0 + i * dx * dx * i));
    }

    if (rank != 0) {

        MPI_Send(&pi_per_rank, 1, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD); 
    }

    if(rank == 0) {

        pi = pi_per_rank;

        for (i = 1; i < size; ++i) {

            MPI_Recv(&pi_per_rank, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);

            pi += pi_per_rank;
        }

        time_used = MPI_Wtime() - start;    
        
        printf("CPU using MPI method time used %lf ms\n", time_used * 1000);\

        pi = 4 * pi;

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