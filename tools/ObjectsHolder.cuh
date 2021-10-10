#pragma once
#include "../drawables/Drawable.cuh"
#include "../drawables/SphereDrawable.cuh"
#include "../drawables/TriangleDrawable.cuh"
#include "../drawables/TriangleMirrorDrawable.cuh"
#include "../drawables/SphereMirrorDrawable.cuh"
#include "../drawables/VoidDrawable.cuh"
#include "../tools/DeviceList.cu"
#include "../lightnings/AmbientLightning.cuh"
#include "../lightnings/DirectLightning.cuh"
#include "../lightnings/PointDirectLightning.cuh"
#include "../lightnings/VoidDirectLightning.cuh"
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
	__device__ Color6Component hit(Ray* r, float tmin, float tmax, int rmax);

	__host__ void setAmbientLight(ObjectsHolder* gpuObj, int* arg);
	__host__ void setSkyBoxColor(ObjectsHolder* gpuObj, int* arg);
	__host__ void setFactor(ObjectsHolder* gpuObj, Vector3D arg);
	__host__ int addDrawable(ObjectsHolder* gpuObj, int* arg);
	__host__ int addLightning(ObjectsHolder* gpuObj, int* arg);
	__host__ int addTexture(ObjectsHolder* gpuObj, unsigned char* tex);
	__host__ int modifyDrawable(ObjectsHolder* gpuObj, int index, int* arg);
	__host__ int modifyLightning(ObjectsHolder* gpuObj, int index, int* arg);
	__host__ int* getDrawables();
	__host__ int getDrawablesSize();
	__host__ int* getLightnings();
	__host__ int getLightningsSize();
	__host__ void syncTextures(ObjectsHolder* gpuObj, int** ptr);

	__host__ void initFromHostSide();
	__device__ void initFromDeviceSide(int* addr);
	__device__ void endFromDevice();

	__device__ Color6Component getLightValueAt(Vector3D* pos, Vector3D* normal);
	__device__ void addDDrawable(int* args);
	__device__ void addDLightning(DirectLightning* arg);
	__device__ void modifyDDrawable(int index, int* arg);
	__device__ void modifyDLightning(int index, DirectLightning* arg);
	__device__ void syncDTextures(int** ptr);

	int dSizeX, dSizeY;
	int dActualSizeX = 2;
	int dActualSizeY = 2;

	DeviceList<Drawable*>* dDrawables;
	DeviceList<DirectLightning*>* dLightnings;
	DeviceList<int*>* dInts;
	AmbientLightning* dAmbientlight;
	Vector3D dFactor;
	Color6Component dSkyBoxColor;

	int* dTestList;

	int dDsize;
	int dLsize;
	int dTsize;

	int hDsize = 0;
	int hTsize = 0;

	const int objectSizeA = 14;
	const int objectSizeB = 8;
private:
	
	int* hDrawables;
	int* hLightnings;
	int** dTextures;

	int hLsize = 0;
	int* hAmbientLight;
	int* hSkyBoxColor;
	Vector3D hFactor;

};