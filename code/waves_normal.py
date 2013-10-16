from numpy import * 
import matplotlib.pyplot as plt
#Define array for results
wav_array = zeros((900,900), dtype=float32)
#Define wave sources
pa = (300,450)
pb = (600,450)
#Loop over all cells in array to calculate values
for i in range(900):
	for j in range(900):
		adist = sin( sqrt((i-pa[0])**2 + (j-pa[1])**2))
		bdist = sin( sqrt((i-pb[0])**2 + (j-pb[1])**2))
		wav_array[i][j] = (adist+bdist)/2
#Show the results
plt.imshow(wav_array)
plt.clim(-1,1);
plt.set_cmap('gray')
plt.axis('off')
plt.show()