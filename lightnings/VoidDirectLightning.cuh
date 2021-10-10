#pragma once
#include "DirectLightning.cuh"
#include "../tools/Color6Component.cuh"

class VoidDirectLightning :public DirectLightning
{
public:
	__device__ VoidDirectLightning();
	__device__ VoidDirectLightning(const DirectLightning& obj);
	__device__ ~VoidDirectLightning();

	__device__ virtual Color6Component getLightByDistance(float dist) override;
	__device__ virtual Vector3D getPos() override;
private:
};