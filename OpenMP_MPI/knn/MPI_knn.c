#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

#define all_num 1000 //总数据的数量
#define test_num 50 //测试数据的数量
#define train_num 950 //训练数据的数量
#define D 10 //特征数据的数量（维数）
#define K 14//K的最大取值
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

				test[n].data[j] = temp[i].data[j]; 
            }
            
			test[n].label = temp[i].label;
			n++;
		}
        else { //剩下的行数存入训练集
            
			for (j = 0; j < D; ++j) {

				train[m].data[j] = temp[i].data[j];
            }
            
			train[m].label = temp[i].label;
			m++;
		}
    }

	fclose(fp);
	fp = NULL;
    
    // printf("test:\n"); //把分割后的数据都打印出来  便于观察是否已经打乱
    
	// for (i = 0; i < test_num; ++i) {

	// 	for (j = 0; j < D; ++j) {

	// 		printf("%lf ", test[i].data[j]);
    //     }
        
	// 	printf("%d\n",test[i].label);
    // }
    
	// printf("\n\ntrain:\n");
    
    // for (i = 0; i < train_num; ++i) {

	// 	for (j = 0; j < D; ++j) {

	// 		printf("%lf ", train[i].data[j]);
    //     }
        
	// 	printf("%d\n",train[i].label);
	// }
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

	// printf("%d %d ", sumA, sumB);
	
	// printf("%d %d \n", maxn(sumA, sumB), test[n].label);
    
    if (maxn(sumA, sumB) == test[n].label) { //检测距离最近的k个标签与原测试标签是否符合  并统计
        
		(*sum)++; //统计符合的数量
	}
}

int cmp(const void* a, const void* b) { //快速排序qsort函数的cmp函数(判断函数)

	Distance A = *(Distance*)a;
	Distance B = *(Distance*)b;
	return A.data > B.data ? 1 : -1;
}

void Normal_Method() {

	int sum = 0;//用于统计距离最近的k个标签与原测试标签符合的数量
	
	int i, j, k;

	// printf("\n\nCheck data:\n\n");

	double result = 0.0;

    double start = omp_get_wtime();

	for (i = 0; i < test_num; i++) {

		for (j = 0; j < train_num; j++) {

			result = 0.0;
			
			for (k = 0; k < D; ++k) {
				
				result += (test[i].data[k] - train[j].data[k]) * (test[i].data[k] - train[j].data[k]);
			}

			distance[j].data = sqrt(result);

			distance[j].trainlabel = train[j].label; //以上距离存入的同时也把训练集标签一起存入distance结构体数组中
        }
            
		qsort(distance, train_num, sizeof(distance[0]), cmp); //用qsort函数从小到大排序测试数据与每组训练数据的距离
        countlabel(&sum, K, i); //统计距离测试集标签最近的标签出现频数
    }

	double time_used = omp_get_wtime() - start;

    printf("CPU parallel using NORMAL method time used %lf ms\n", time_used * 1000);

	printf("K = %d     P = %.1lf%%\n", K, 100.0 * (sum) / test_num); //打印K值对应的概率
}

void MPI(int argc, char *argv[]) {

	int sum = 0, sum_per_rank = 0;//用于统计距离最近的k个标签与原测试标签符合的数量
	
	int i, j, k, rank, size;

	// printf("\n\nCheck data:\n\n");

	double result = 0.0, result_per_rank = 0.0, start, time_used;

    MPI_Status status;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {

		Normal_Method();

        start = MPI_Wtime();
    }

	for (i = 0; i < test_num; i++) {

		for (j = 0; j < train_num; j++) {

			result_per_rank = result = 0.0;
			
			for (k = rank; k < D; k += size) {

				result_per_rank += (test[i].data[k] - train[j].data[k]) * (test[i].data[k] - train[j].data[k]);
			}

			if (rank != 0) {

				MPI_Send(&result_per_rank, 1, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD); 
			}

			if(rank == 0) {

				result = result_per_rank;

				for (k = 1; k < size; ++k) {

					MPI_Recv(&result_per_rank, 1, MPI_DOUBLE, k, k, MPI_COMM_WORLD, &status);

					result += result_per_rank;
				}

				distance[j].data = sqrt(result);

				distance[j].trainlabel = train[j].label; //以上距离存入的同时也把训练集标签一起存入distance结构体数组中
			}
        }

		if (rank == 0) {
            
			qsort(distance, train_num, sizeof(distance[0]), cmp); //用qsort函数从小到大排序测试数据与每组训练数据的距离
			countlabel(&sum, K, i); //统计距离测试集标签最近的标签出现频数
		}
    }

	if (rank == 0) {

		time_used = MPI_Wtime() - start;

		printf("CPU using MPI method time used %lf ms\n", time_used * 1000);

		printf("K = %d     P = %.1lf%%\n", K, 100.0 * (sum) / test_num); //打印K值对应的概率
	}

	MPI_Finalize();
}

int main(int argc, char *argv[]) {

    loaddata("targetclass.txt");

	MPI(argc, argv);

	return 0;
}
