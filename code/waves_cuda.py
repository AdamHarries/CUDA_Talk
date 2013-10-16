from numpy import * 
import matplotlib.pyplot as plt
import pycuda.compiler as comp
import pycuda.driver as drv
import pycuda.autoinit

#Define the CUDA kernel, and compile it
mod = comp.SourceModule("""
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
""")

#Set up the array in normal memory
wav_array = zeros((900,900), dtype=float32)
#Set up the array on the GPU
wav_array_gpu = drv.mem_alloc(wav_array.nbytes)
#get kernel function ready to run
make_waves = mod.get_function("sin_dist")
#runs the kernel
make_waves(wav_array_gpu,block=(30,30,1), grid=(30,30))
#Compy the resluts back
drv.memcpy_dtoh(wav_array, wav_array_gpu)
#Displays the results
plt.imshow(wav_array)
plt.clim(-1,1);
plt.set_cmap('gray')
plt.axis('off')
plt.show()