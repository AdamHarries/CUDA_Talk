/*compile using: gcc -std=c99 waves_normal.c -o wn*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//Main function
int main(int argc, char** argv)
{
	//Allocate space for wave array
	float *wav_array = malloc(sizeof(float)*900*900);
	//define wave sources
	float pa[] = {100, 150};
	float pb[] = {200, 150};
	int index = 0;
	//Loop over all cells in array
	for(int i = 0;i<900;i++){
		for(int j = 0;j<900;j++){
			//calculate 1d index from 2d index
			index = i+(j*900);
			//calculate value at cell
			float adist = sqrt( ((i-pa[0])*(i-pa[0]))+((j-pa[1])*(j-pa[1])) );
			float bdist = sqrt( ((i-pb[0])*(i-pb[0]))+((j-pb[1])*(j-pb[1])) );
			//set value in cell
			wav_array[index] = (sin(adist)+sin(bdist))/2;		
		}
	}
	return 0;
}