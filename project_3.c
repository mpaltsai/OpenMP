#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>

double gettime(void)
{
	struct timeval ttime;
	gettimeofday(&ttime , NULL);
	return ttime.tv_sec + ttime.tv_usec * 0.000001;
}

int main(int argc, char ** argv)
{
	double timeTotal = 0.0f;
	double time0=gettime();
	
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int l = atoi(argv[3]);
	
	unsigned long long checksum = 0; //a variable to store the sum of distances
	
	
	//initialize a seed to generate the series for random values of mset and nset of strings with function rand()
	srand(0);
	
	//memory allocation for m,n sets	
	int ** mset = (int**)malloc(sizeof(int*)*m);
	assert(mset!=NULL);
	
	int ** nset = (int**)malloc(sizeof(int*)*m);
	assert(nset!=NULL);
	
	//memory allocation for a m*n matrix so as to store Hamming distances from string comparison
	unsigned long long ** mn = (unsigned long long**)malloc(sizeof(unsigned long long*)*m);
	assert(mn!=NULL);

	//initialize strings in mset & nset
	for(int i=0; i<m; i++){
		//allocate memory for each string
		mset[i] = (int*)malloc(sizeof(int)*l);
		nset[i] = (int*)malloc(sizeof(int)*l);
		
		for(int j=0; j<l; j++){
			//initialize with random values
			mset[i][j] = (int)(rand()%2);
			nset[i][j] = (int)(rand()%2);

		}
		
	}
	
	//pairwise string comparison to compute Hamming distance and write it to mn matrix
	for(int i=0; i<m; i++){

		mn[i] = (unsigned long long*)malloc(sizeof(unsigned long long)*m);  //allocate memory for the columns of the mn matrix
		
		for(int j=0; j<n; j++){
			
			unsigned long long d = 0; //a variable to store Hamming distance
			
			for(int k=0; k<l; k++){

				if(mset[i][k] != nset[j][k])
					d += 1;
				else
					d += 0;				
			}
			
			//write the distance to the appropriate position in mn matrix
			mn[i][j] = d; 
			
		}
		
	}
	
	//calculate the sum of Hamming distances stored in mn matrix
	for(int i=0; i<n; i++){

		for(int j=0; j<m; j++){
			
			checksum += mn[i][j];
			
		}
		
	}
	
	printf("Checksum is %lld\n", checksum);
	
	double time1=gettime();
	timeTotal += time1-time0;
	
	printf("Time %f\n", timeTotal);
	
	free(mset);
	free(nset);
	free(mn);
}
