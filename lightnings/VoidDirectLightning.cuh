#pragma once
#include "DirectLightning.cuh"
#include "Color6Component.cuh"

class VoidDirectLightning :public DirectLightning
{
public:
	__device__ VoidDirectLightning();
	__device__ VoidDirectLightning(const DirectLightning& obj);
	__device__ ~VoidDirectLightning();

	__device__ virtual Color6Component getLightByDistance(double dist) override;
	__device__ virtual VectorDouble getPos() override;
private:
};