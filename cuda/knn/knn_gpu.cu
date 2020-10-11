#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>

#define all_num 1000 //总数据的数量
#define test_num 50 //测试数据的数量
#define train_num 950 //训练数据的数量
#define D 10 //特征数据的数量（维数）
#define K 14//K的最大取值
#define THREAD_NUM 2
#define BLOCK_NUM D / 2

typedef struct {

	double data;//距离
	int trainlabel;//用于链接训练标签
}Distance;

typedef struct {

	double data[D];//特征数据
	int label; //标签
}data;

data test[test_num];//测试数据结构体数组
data train[train_num];//训练数据结构体数组
data temp[all_num]; //临时存放数据结构体数组
Distance distance[train_num];//存放距离结构体数组

void loaddata(char filename[]) { //加载数据      分割：测试test_num组   训练train_num组

	FILE* fp = NULL;
	fp = fopen(filename, "r");
	int i, j;
	int n = 0, m = 0;

	for (i = 0; i < all_num; ++i) {

		for (j = 0; j < D; ++j) {

			fscanf(fp, "%lf ", &temp[i].data[j]);
        }
        
        fscanf(fp, "%d", &temp[i].label);
	}
	
	srand((unsigned int)time(NULL));

	for (i = 0; i < all_num; ++i) {

		int n1 = (rand() % all_num);//产生n以内的随机数  n是数组元素个数
        int n2 = (rand() % all_num);
        
        if (n1 != n2) { //若两随机数不相等 则下标为这两随机数的数组进行交换
            
			data t = temp[n1];
			temp[n1] = temp[n2];
			temp[n2] = t;
		}
	}

	for (i = 0; i < all_num; i++) {

        if (i < test_num) { //存入测试集
            
			for (j = 0; j < D; j++) {

				test[n].data[j] = temp[i].data[j]; //存入花的四个特征数据
            }
            
			test[n].label = temp[i].label;
			n++;
		}
        else { //剩下的行数存入训练集
            
			for (j = 0; j < D; ++j) {

				train[m].data[j] = temp[i].data[j];//存入花的四个特征数据
            }
            
			train[m].label = temp[i].label;
			m++;
		}
    }

	fclose(fp);
	fp = NULL;
    
    printf("test:\n"); //把分割后的数据都打印出来  便于观察是否已经打乱
    
	for (i = 0; i < test_num; ++i) {

		for (j = 0; j < D; ++j) {

			printf("%lf ", test[i].data[j]);
        }
        
		printf("%d\n",test[i].label);
    }
    
	printf("\n\ntrain:\n");
    
    for (i = 0; i < train_num; ++i) {

		for (j = 0; j < D; ++j) {

			printf("%lf ", train[i].data[j]);
        }
        
		printf("%d\n",train[i].label);
	}
}

__global__ static void computedistance(double *gpu_train, double *gpu_test, double *gpu_dist) { //计算距离

	double sum = 0.0;
	
	for (int i = blockIdx.x * THREAD_NUM + threadIdx.x; i < D; i += BLOCK_NUM * THREAD_NUM) {

		sum += (gpu_test[i] - gpu_train[i]) * (gpu_test[i] - gpu_train[i]);
    }
    
	gpu_dist[blockIdx.x * THREAD_NUM + threadIdx.x] = sum;
}

int maxn(int a, int b) { //找出频数最高的 测试数据就属于出现次数最高的

	if (a > b) return 0;

	else if(b > a) return 1;

	return 0;	
}

void countlabel(int* sum ,int k, int n) { //统计距离最邻近的k个标签出现的频数

	int i;
    int sumA = 0, sumB = 0; 
    
	for (i = 0; i < k; ++i) {

        switch (distance[i].trainlabel) { //用Distance结构体指针p来取K个距离最近的标签来进行判断
						
            case 0: sumA++; break;
            case 1: sumB++; break;
		}
	}

	printf("%d %d ", sumA, sumB);
	
	printf("%d %d \n", maxn(sumA, sumB), test[n].label);
    
    if (maxn(sumA, sumB) == test[n].label) { //检测距离最近的k个标签与原测试标签是否符合  并统计
        
		(*sum)++; //统计符合的数量
	}
}

int cmp(const void* a, const void* b) { //快速排序qsort函数的cmp函数(判断函数)

	Distance A = *(Distance*)a;
	Distance B = *(Distance*)b;
	return A.data > B.data ? 1 : -1;
}

int main() {

    loaddata("targetclass.txt");
    
    int sum = 0;//用于统计距离最近的k个标签与原测试标签符合的数量
	
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	printf("\n\nCheck data:\n\n");

	double *gpu_test, *gpu_train, *gpu_dist;

	double dist[D], result;

	cudaMalloc((void**)&gpu_test, sizeof(double) * D);
    cudaMalloc((void**)&gpu_train, sizeof(double) * D);
	cudaMalloc((void**)&gpu_dist, sizeof(double) * D);
	
	for (int i = 0; i < test_num; i++) {

		for (int j = 0; j < train_num; j++) {

			result = 0;

            cudaMemcpy(gpu_test, test[i].data, sizeof(double) * D, cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_train, train[j].data, sizeof(double) * D, cudaMemcpyHostToDevice);

            computedistance << <BLOCK_NUM, THREAD_NUM, 0>> >(gpu_train, gpu_test, gpu_dist);

			cudaMemcpy(dist, gpu_dist, sizeof(double) * D, cudaMemcpyDeviceToHost);
				
			for (int k = 0; k < D; ++k) {

				result += dist[k];
			}

			distance[j].data = sqrt(result);

			distance[j].trainlabel = train[j].label; //以上距离存入的同时也把训练集标签一起存入distance结构体数组中
        }
            
		qsort(distance, train_num, sizeof(distance[0]), cmp); //用qsort函数从小到大排序测试数据与每组训练数据的距离
        countlabel(&sum, K, i); //统计距离测试集标签最近的标签出现频数
    }
        
	printf("K = %d     P = %.1lf%%\n", K, 100.0 * (sum) / test_num); //打印K值对应的概率
	cudaFree(gpu_test);
	cudaFree(gpu_train);
	cudaFree(gpu_dist);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
				
	printf("GPU_time: %f ms\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}
