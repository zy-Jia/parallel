#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//CUDA RunTime API
#include <cuda_runtime.h>

//thread代表一次运算，通过指定THREAD_NUM来规定要有多少个线程并行执行，从而优化程序执行所需时间
#define THREAD_NUM 1024
#define MATRIX_SIZE 2000

//__global__函数（GPU上执行）计算矩阵乘法
__global__ static void matrixMult(const double *a, const double *b, double *c, int n) {

    //表示目前的 thread 是第几个 thread（由 0 开始计算）
    const int tid = threadIdx.x;

    //表示目前的 thread 属于第几个 block（由 0 开始计算）
    const int bid = blockIdx.x;

    //从 bid 和 tid 计算出这个 thread 应该计算的 row 和 column
    const int idx = bid * THREAD_NUM + tid;
    const int row = idx / n;
    const int column = idx % n;

    //矩阵乘法
    if(row < n && column < n) {

        double  t = 0;

        for (int i = 0; i < n; ++i) {

            t += a[row * n + i] * b[i * n + column];
        }

        c[row * n + column] = t;
    }
}

void generateMatrix(double *a, int n) {

    for (int i = 0; i < n * n; ++i) {

        a[i] = 1.0 * rand() / RAND_MAX;    
    }
}

int main() {

    int blockSize;
    int minGridSize;
    int gridSize;

    cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    //定义矩阵
    double *a, *b, *c, *d;
    int n = MATRIX_SIZE;

    //分配内存
    a = (double*)malloc(sizeof(double) * n * n);
    b = (double*)malloc(sizeof(double) * n * n);
    c = (double*)malloc(sizeof(double) * n * n);
    d = (double*)malloc(sizeof(double) * n * n);

    //设置随机数种子
    srand((unsigned int)time(NULL));

    //生成随机矩阵
    generateMatrix(a, n);
    generateMatrix(b, n);

    /*把数据复制到显卡内存中*/
    double *gpu_a, *gpu_b, *gpu_c;

    //cudaMalloc取得一块显卡内存
    cudaMalloc((void**)&gpu_a, sizeof(double) * n * n);
    cudaMalloc((void**)&gpu_b, sizeof(double) * n * n);
    cudaMalloc((void**)&gpu_c, sizeof(double) * n * n);

    //cudaMemcpy将产生的随机数复制到显卡内存中
    //cudaMemcpyHostToDevice：从内存复制到显卡内存
    //cudaMemcpyDeviceToHost：从显卡内存复制到内存
    cudaMemcpy(gpu_a, a, sizeof(double) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, sizeof(double) * n * n, cudaMemcpyHostToDevice);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMult, 0, n * n); 
    gridSize = (n * n + blockSize - 1) / blockSize; 

    cudaEventRecord(start, 0);

    //在CUDA中执行函数 语法：函数名称<<<block数目，thread数目，shared memory大小>>>（参数...）；
    matrixMult << <gridSize, blockSize, 0>> >(gpu_a, gpu_b, gpu_c, n);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
	printf("GPUtime: %f ms\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Blocksize %i\n", blockSize);
    printf("Gridsize %i\n", gridSize);

    /*把结果从显示芯片复制回主内存*/
    //cudaMemcpy将结果从显存中复制回内存
    cudaMemcpy(c, gpu_c, sizeof(double) * n * n, cudaMemcpyDeviceToHost);

    //cudaFree
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);

    clock_t cpu_st = clock();

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) { 

            double t = 0;

            for (int k = 0; k < n; k++) { 

                t += a[i * n + k] * b[k * n + j]; 
            } 

            d[i * n + j] = t; 
        } 
    }

    clock_t cpu_et = clock() - cpu_st;

    printf("CPUtime: %lf ms\n", (double)(cpu_et)/CLOCKS_PER_SEC * 1000);

    //验证正确性与精确性
    double max_err = 0;

    double average_err = 0; 

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {

            if (d[i * n + j] != 0) { 

                //fabs求浮点数x的绝对值
                double err = fabs((c[i * n + j] - d[i * n + j]) / d[i * n + j]);

                if (max_err < err) max_err = err; 

                average_err += err; 
            } 
        } 
    }

    printf("Max error: %g Average error: %g\n", max_err, average_err / (n * n));

    return 0;
}


