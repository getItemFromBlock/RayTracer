#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ObjectsHolder.cuh"
#include "Camera.cuh"

ObjectsHolder* mainHolder;
ObjectsHolder* deviceHolder;
Camera* camera;
bool isInitialised = false;

__constant__ int dtestL[13 * 50];

constexpr int M = 512;

Vector3D cameraPos = Vector3D(1, 5.0, 0);
Vector3D cameraTarget = Vector3D(2, 0.5, 0);
Vector3D cameraRot = Vector3D(0, 1, 0);
float cameraFOV = 90;

__global__ void setGPUResolution(ObjectsHolder* holder, int newX, int newY) {
	holder->dSizeX = newX;
	holder->dSizeY = newY;
}

__global__ void getGPUResolution(ObjectsHolder* holder, int* res) {
	res[0] = holder->dActualSizeX;
	res[1] = holder->dActualSizeY;
}

__global__ void initGPUSide(ObjectsHolder* gpuObj, Camera* cameraObj) {
	*gpuObj = ObjectsHolder();
	gpuObj->initFromDeviceSide(dtestL);
	*cameraObj = Camera(Vector3D(2, 0, 0), Vector3D(1, -0.5, 0), Vector3D(0, 1, 0), 90, 16.0 / 9.0);
	cameraObj->refresh(Vector3D(2, 0, 0), Vector3D(1, -0.5, 0), Vector3D(0, 1, 0), 90, 16.0 / 9.0);
}

__global__ void mainRender(ObjectsHolder* holder, Camera* camObj, int* outputScreen) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index + 1 <= (holder->dActualSizeX*holder->dActualSizeY))
	{
		int i = index % holder->dActualSizeX;
		int j = index / holder->dActualSizeX;
		float u = (0.5 + i) / holder->dActualSizeX;
		float v = 1 - (0.5 + j) / holder->dActualSizeY;
		Ray r = camObj->getRay(u, v);

		Color6Component pixelColor = holder->hit(&r, 0.00001, 10000.0, 20);
		outputScreen[index] = pixelColor.getRGBComponent();
	}
	if (index == 0) {
		holder->dActualSizeX = holder->dSizeX;
		holder->dActualSizeY = holder->dSizeY;
	}
	return;
}

__global__ void endGPUSide(ObjectsHolder* holder, Camera* cameraObj) {
	delete cameraObj;
	holder->endFromDevice();
	delete holder;
}

__global__ void refreshGPUCamera(Camera* camObj, float* args) {
	camObj->refresh(Vector3D(args[0], args[1], args[2]), Vector3D(args[3], args[4], args[5]), Vector3D(args[6], args[7], args[8]), args[9], args[10]);
}

__host__ void refreshCamera(float ratio) {
	if (isInitialised && camera) {
		float* gpuArgs;
		gpuErrchk(cudaMalloc((void**)&gpuArgs, 11 * sizeof(float)));
		float* rawArgs = (float*)malloc(11 * sizeof(float));
		rawArgs[0] = cameraPos.getX();
		rawArgs[1] = cameraPos.getY();
		rawArgs[2] = cameraPos.getZ();
		rawArgs[3] = cameraTarget.getX();
		rawArgs[4] = cameraTarget.getY();
		rawArgs[5] = cameraTarget.getZ();
		rawArgs[6] = cameraRot.getX();
		rawArgs[7] = cameraRot.getY();
		rawArgs[8] = cameraRot.getZ();
		rawArgs[9] = cameraFOV;
		rawArgs[10] = ratio;
		gpuErrchk(cudaMemcpy(gpuArgs, rawArgs, 11 * sizeof(float), cudaMemcpyHostToDevice));
		Camera* tempObjC = camera;
		refreshGPUCamera << <1, 1 >> > (tempObjC, gpuArgs);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaFree(gpuArgs));
	}
}

__host__ void setResolution(int x, int y)
{
	ObjectsHolder* tempObjH = deviceHolder;
	setGPUResolution << <1, 1 >> > (tempObjH, x, y);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

__host__ int addTexture(unsigned char*  tex) {
	ObjectsHolder* tempObjH = deviceHolder;
	return mainHolder->addTexture(tempObjH, tex);
}

__host__ int addDrawableObj(int* arg) {
	gpuErrchk(cudaMemcpyToSymbol(dtestL, arg, 13 * sizeof(int), mainHolder->hDsize * 13 * sizeof(int), cudaMemcpyHostToDevice));
	ObjectsHolder* tempObjH = deviceHolder;
	return mainHolder->addDrawable(tempObjH, arg);
}

__host__ int modDrawableObj(int index, int* arg) {
	gpuErrchk(cudaMemcpyToSymbol(dtestL, arg, 13 * sizeof(int), mainHolder->hDsize * 13 * sizeof(int), cudaMemcpyHostToDevice));
	ObjectsHolder* tempObjH = deviceHolder;
	return mainHolder->modifyDrawable(tempObjH, index, arg);
}

__host__ void setSkyBoxColorRGB(int* arg) {
	ObjectsHolder* tempObjH = deviceHolder;
	mainHolder->setSkyBoxColor(tempObjH, arg);
	return;
}

__host__ int addLightningObj(int* arg) {
	ObjectsHolder* tempObjH = deviceHolder;
	return mainHolder->addLightning(tempObjH, arg);
}

__host__ void setInitialised() {
	isInitialised = true;
}

__host__ void initRayTracer(ObjectsHolder* obj) {
	mainHolder = obj;
	mainHolder->initFromHostSide();
	mainHolder->dTestList = dtestL;
	gpuErrchk(cudaMalloc((void**)&deviceHolder, sizeof(ObjectsHolder)));
	ObjectsHolder* tempObjH = deviceHolder;
	gpuErrchk(cudaMalloc((void**)&camera, sizeof(Camera)));
	Camera* tempObjC = camera;
	initGPUSide << <1, 1 >> > (tempObjH, tempObjC);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

__host__ int* renderImage() {
	if (isInitialised) {
		int* gpuRes;
		gpuErrchk(cudaMalloc((void**)&gpuRes, 2 * sizeof(int)));
		ObjectsHolder* tempObjH = deviceHolder;
		getGPUResolution << < 1, 1 >> > (tempObjH, gpuRes);
		gpuErrchk(cudaPeekAtLastError());

		int* imageRes = (int*)malloc(2 * sizeof(int));
		gpuErrchk(cudaMemcpy(imageRes, gpuRes, 2 * sizeof(int), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaFree(gpuRes));
		if (imageRes && imageRes[1] != 0) {
			refreshCamera(((float)imageRes[0]) / ((float)imageRes[1]));
		}
		else {
			refreshCamera(16.0 / 9.0);
		}
		int N = imageRes[0] * imageRes[1];
		int* image = (int*)malloc((N + 2) * sizeof(int));
		int* gpuImage;
		gpuErrchk(cudaMalloc((void**)&gpuImage, N * sizeof(int)));
		ObjectsHolder* tempObjH2 = deviceHolder;
		Camera* tempObjC = camera;
		mainRender << <(N + M - 1) / M, M >> > (tempObjH2, tempObjC, gpuImage);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(image + 2, gpuImage, N * sizeof(int), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaFree(gpuImage));
		image[0] = imageRes[0];
		image[1] = imageRes[1];
		return image;
	}
	return nullptr;
}

__host__ void endRayTracer() {
	delete mainHolder;
	ObjectsHolder* tempObjH = deviceHolder;
	Camera* tempObjC = camera;
	endGPUSide << < 1, 1 >> > (tempObjH, tempObjC);
}

__host__ void setCameraArgs(Vector3D arg1, Vector3D arg2, Vector3D arg3, float arg4)
{
	cameraPos = arg1;
	cameraTarget = arg2;
	cameraRot = arg3;
	cameraFOV = arg4;
}