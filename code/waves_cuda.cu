#include <stdio.h>
#include <stdlib.h>


/*wave kernel*/
__global__ void sin_dist(float *wa)
{
	/*calculate 2d arrray index from thread and block IDs*/
	const int j = threadIdx.y+(blockIdx.y*gridDim.y);
	const int i = threadIdx.x+(blockIdx.x*gridDim.x);
	/*calculate mapping to 1d array from 2d indicies*/
	const int lID = i+(j*blockDim.y*gridDim.y);
	/*define the wave sources*/
	float pa[] = {100,150};
	float pb[] = {200,150};
	/*calculate value at location*/
	float adist = sqrt( ((i-pa[0])*(i-pa[0]))+((j-pa[1])*(j-pa[1])) );
	float bdist = sqrt( ((i-pb[0])*(i-pb[0]))+((j-pb[1])*(j-pb[1])) );
	/*save to array*/
	wa[lID] = (sin(adist)+sin(bdist))/2;		
}

/*Main function (ofc)*/
int main(int argc, char** argv)
{
	/*allocate memory for "normal" array*/
	float *wav_array = (float*)malloc(sizeof(float)*900*900);
	/*allocate memory on the GPU*/
	float *gpu_wav_array;
	cudaMalloc(&gpu_wav_array, 900*900*sizeof(float));
	/*define block size*/
	dim3 block_size;
  	block_size.x = 30;
  	block_size.y = 30;
  	block_size.z = 1;
  	/*define grid dimensions*/
  	dim3 grid_size;
  	grid_size.x = 30;
  	grid_size.y = 30;
  	/*Launch the kernels*/
	sin_dist<<<block_size, grid_size>>>(gpu_wav_array);
	/*Copy the data back from the GPU*/
	cudaMemcpy(wav_array, gpu_wav_array, 900*900*sizeof(float), cudaMemcpyDeviceToHost);
	return 0;
}