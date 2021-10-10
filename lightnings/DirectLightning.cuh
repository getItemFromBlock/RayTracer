#pragma once
#include "../tools/Vector3D.cuh"
#include "../tools/Color6Component.cuh"

class DirectLightning
{
public:
	__device__ DirectLightning();
	__device__ DirectLightning(const DirectLightning& obj);
	__device__ virtual Color6Component getLightByDistance(float dist) = 0;
	__device__ virtual Vector3D getPos() = 0;
};