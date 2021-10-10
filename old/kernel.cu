#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <device_launch_parameters.h>
#include "Header.h"
#include "VectorDouble.cuh"
#include "SphereDrawable.cuh"
#include "Drawable.cuh"
#include "ObjectsHolder.cuh"

constexpr int N = (2000);		// Nombre total de valeurs
constexpr int M = 512;				// Nombre de Threads par Blocks

__global__ void addSphereDrawable(ObjectsHolder *list, int* c, double posX, double posY, double posZ, double radius, double colorR, double colorG, double colorB)
{
	SphereDrawable* obj = new SphereDrawable(VectorDouble(posX, posY, posZ), radius, VectorDouble(colorR, colorG, colorB));
	list->add(obj);
	c[1] = list->get_l(0)->getPos().getY();
}

__global__ void addPointDirectLightning(ObjectsHolder *list, int* c, double posX, double posY, double posZ, double colorR, double colorG, double colorB, double atenuation)
{
	PointDirectLightning* obj = new PointDirectLightning(VectorDouble(posX, posY, posZ), VectorDouble(colorR, colorG, colorB), atenuation);
	list->addDirectLightning(obj);
	c[0] = obj->getPos().getY();
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}



__global__ void init_kernel(ObjectsHolder *list, VectorDouble* d_factor, VectorDouble* d_skyBoxColor, int *d_size, int *d_Asize, VectorDouble* temp_Alight) {
	VectorDouble* test = new VectorDouble();
	*test = VectorDouble(5, 3, 2);
	*list = ObjectsHolder();
	(list)->setFactorAdress(d_factor);
	(list)->setSkyBoxAdress(d_skyBoxColor);
	(list)->setDSizeAdress(d_size);
	(list)->setLSizeAdress(d_Asize);
	(list)->setAmbientLightAdress(temp_Alight);
	(list)->initFromDeviceSide();
}

__global__ void end_kernel(ObjectsHolder* list) {
	list->endFromDevice();
}

// kernel code, s'execute au niveau du Device (GPU et vRAM)
__global__ void main_kernel(int* c, ObjectsHolder *list, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n)
	{
		c[index] = list->get_d(1)->getColor(new HitRecord())->getX();
	}
}

void drawScene()
{

}

void fill_ints(int* x, int n, int param1, int param2) {
	for (int i = 0; i < n; i++) {
		x[i] = param1 * i + param2;
	}
}

// Code principal, s'execute au niveau du Host (CPU et RAM)
namespace Wrapper {
	void wrapper(void) {

		
		int* c;
		int* d_c;			// Variables au niveau vRAM
		int size = N * sizeof(int);		// Taille des variables pour l'allocation

		// Alloue l'espace mémoire nécessaire dans la vRAM
		gpuErrchk(cudaMalloc((void**)&d_c, size));
		ObjectsHolder* d_o;
		// Alloue l'espace mémoire nécessaire dans la RAM et remplis les valeurs a et b
		c = (int*)malloc(size);
		

		// Copie les valeurs de a et b dans la vRAM
		ObjectsHolder* o = new ObjectsHolder();
		o->setAmbientLight(new VectorDouble(5, 5, 5));
		gpuErrchk((cudaMalloc((void**)&d_o, sizeof(ObjectsHolder))));
		o->initFromHostSide();
		
		VectorDouble* d_factor = o->getFactorAdress();
		VectorDouble* d_skyBoxColor = o->getSkyBoxAdress();
		int *d_size = o->getDSizeAdress();
		int *d_Asize = o->getLSizeAdress();
		VectorDouble* temp_Alight = o->getAmbientLightAdress();

		ObjectsHolder* zertygsdfgsd = d_o;
		init_kernel<<<1, 1>>>(zertygsdfgsd, d_factor, d_skyBoxColor, d_size, d_Asize, temp_Alight);

		constexpr double t = 120;
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		ObjectsHolder* reyrdtgzer = d_o;
		addPointDirectLightning<<<1, 1>>>(reyrdtgzer, d_c, 0.0, 5.0, 0.0, t, t, t, 98.0);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		addSphereDrawable<<<1, 1 >>>(zertygsdfgsd, d_c, 0.0, 0.0, 0.0, 1.0, t, 3.0, 8.0);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));
		printf(" %i \n", c[0]);
		printf(" %i \n", c[1]);

		addSphereDrawable<<<1, 1 >>>(zertygsdfgsd, d_c, 1.0, 1.0, 1.0, 0.5, 33.0, 0.0, t);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		// Lance le kernel sur le GPU avec (N / THREADS_PER_BLOCK) Block et THREADS_PER_BLOCK Threads
		ObjectsHolder* uybjgnqrt = d_o;
		main_kernel<<<(N + M - 1) / M, M >>>(d_c, uybjgnqrt, N);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		
		ObjectsHolder* ygghsrtrdfg = d_o;
		end_kernel<<<1, 1>>>(ygghsrtrdfg);
		cudaDeviceSynchronize();
		// Copie le résultat de c dans la RAM
		gpuErrchk(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));
		gpuErrchk(cudaFree(d_o));
		getRender(c);
		
		delete o;
		//Envoie les informations au script main
		
		
		
		// Libère la mémoire
		free(c);
		cudaFree(d_c);
	}
}