#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

#define THREAD_NUM 1024

using namespace cv;
using namespace std;

__global__ static void rgb2gray(int rows, int cols, int *gpu_r, int *gpu_g, int *gpu_b, int *gpu_gray) {

    const int idx = blockIdx.x * THREAD_NUM + threadIdx.x;
    const int row = idx / cols;
    const int column = idx % cols;

    //矩阵乘法
    if(row < rows && column < cols) {

		gpu_gray[idx] = gpu_r[idx] * 0.299 + gpu_g[idx] * 0.587 + gpu_b[idx] * 0.114;
	}
}

int main() {

	Mat src = imread("rgb.jpg");
	
	Mat grayMat(src.rows, src.cols, CV_8UC1, Scalar(0));

	vector<Mat> rgbChannels(3);
	Mat rChannel, gChannel, bChannel;
	
	split(src, rgbChannels);
	rChannel = rgbChannels[0];
	gChannel = rgbChannels[1];
	bChannel = rgbChannels[2];

	int *r = new int[src.rows * src.cols];
	int *g = new int[src.rows * src.cols];
	int *b = new int[src.rows * src.cols];
	int *gray = new int[src.rows * src.cols];

	int n = 0;

	for (int i = 0; i < src.rows; ++i) {

		for (int j = 0; j < src.cols; ++j) {

			r[n] = rChannel.at<uchar>(i, j);
			g[n] = gChannel.at<uchar>(i, j);
			b[n] = bChannel.at<uchar>(i, j);
			n += 1;
		}
	}
	
    int *gpu_r, *gpu_g, *gpu_b, *gpu_gray;

    cudaMalloc((void**)&gpu_r, sizeof(int) * src.rows * src.cols);
    cudaMalloc((void**)&gpu_g, sizeof(int) * src.rows * src.cols);
	cudaMalloc((void**)&gpu_b, sizeof(int) * src.rows * src.cols);
	cudaMalloc((void**)&gpu_gray, sizeof(int) * src.rows * src.cols);
	
	cudaMemcpy(gpu_r, r, sizeof(int) * src.rows * src.cols, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_g, g, sizeof(int) * src.rows * src.cols, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, b, sizeof(int) * src.rows * src.cols, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	rgb2gray << <(src.rows * src.cols + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM, 0>> >(src.rows, src.cols, gpu_r, gpu_g, gpu_b, gpu_gray);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
				
	printf("GPUtime: %f ms\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(gray, gpu_gray, sizeof(int) * src.rows * src.cols, cudaMemcpyDeviceToHost);

	cudaFree(gpu_r);
	cudaFree(gpu_g);
	cudaFree(gpu_b);
	cudaFree(gpu_gray);

	n = 0;

	for (int i = 0; i < src.rows; ++i) {

		for (int j = 0; j < src.cols; ++j) {

			grayMat.at<uchar>(i, j) = gray[n];
			n += 1;
		}
	}
	
	Mat merge_img;

	vector<Mat> channels_all;
	channels_all.push_back(grayMat);

	merge(channels_all, merge_img);

	imwrite("gray.jpg", merge_img);

	Mat cpu_gray(src.rows, src.cols, CV_8UC1, Scalar(0));

	clock_t cpu_st = clock();

	for (int y = 0; y < src.rows; y++) {

			  uchar*cp = src.ptr<uchar>(y);
			  uchar*gp = cpu_gray.ptr<uchar>(y);
			  for (int x = 0; x < src.cols; x++) {

					   *gp = (15*cp[0] + 75*cp[1] + 38*cp[2]) >> 7;
					   cp += 3;
					   gp++;
			  }
	}

	clock_t cpu_et = clock() - cpu_st;

    printf("CPUtime: %lf ms\n", (double)(cpu_et)/CLOCKS_PER_SEC * 1000);

	imwrite("cpu_gray.jpg", cpu_gray);

    return 0;
}