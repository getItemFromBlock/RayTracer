#pragma once
#include "Drawable.cuh"
#include "SphereDrawable.cuh"
#include "TriangleDrawable.cuh"
#include "TriangleMirrorDrawable.cuh"
#include "VoidDrawable.cuh"
#include "DeviceList.cu"
#include "AmbientLightning.cuh"
#include "DirectLightning.cuh"
#include "PointDirectLightning.cuh"
#include "VoidDirectLightning.cuh"
#include "iostream"
#include "vector"
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

class ObjectsHolder
{
public:
	__host__ __device__ ObjectsHolder();	// Constructeur - déclaration
	__host__ __device__ ~ObjectsHolder();		// Destructeur - déclaration
	__host__ __device__ ObjectsHolder& operator=(const ObjectsHolder&);
	__host__ __device__ ObjectsHolder(const ObjectsHolder&);

	__device__ Drawable* get_d(unsigned int index);
	__device__ DirectLightning* get_l(unsigned int index);
	__device__ Color6Component hit(Ray* r, double tmin, double tmax, int rmax);

	__host__ void setAmbientLight(ObjectsHolder* gpuObj, int* arg);
	__host__ void setSkyBoxColor(ObjectsHolder* gpuObj, int* arg);
	__host__ void setFactor(ObjectsHolder* gpuObj, VectorDouble arg);
	__host__ int addDrawable(ObjectsHolder* gpuObj, int* arg);
	__host__ int addLightning(ObjectsHolder* gpuObj, int* arg);
	__host__ int modifyDrawable(ObjectsHolder* gpuObj, int index, int* arg);
	__host__ int modifyLightning(ObjectsHolder* gpuObj, int index, int* arg);

	__host__ void initFromHostSide();
	__device__ void initFromDeviceSide();
	__device__ void endFromDevice();

	__device__ Color6Component getLightValueAt(VectorDouble* pos);
	__device__ void addDDrawable(int* args);
	__device__ void addDLightning(DirectLightning* arg);
	__device__ void modifyDDrawable(int index, int* arg);
	__device__ void modifyDLightning(int index, DirectLightning* arg);

	int dSizeX, dSizeY;
	int dActualSizeX = 2;
	int dActualSizeY = 2;

	DeviceList<Drawable*>* dDrawables;
	DeviceList<DirectLightning*>* dLightnings;
	DeviceList<int*>* dInts;
	AmbientLightning* dAmbientlight;
	VectorDouble dFactor;
	Color6Component dSkyBoxColor;

	int dDsize;
	int dLsize;
private:

	

	const int objectSizeA = 13;
	const int objectSizeB = 8;
	int* hDrawables;
	int* hLightnings;
	int hDsize = 0;
	int hLsize = 0;
	int* hAmbientLight;
	int* hSkyBoxColor;
	VectorDouble hFactor;

};