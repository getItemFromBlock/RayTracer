#pragma once
#include "VectorDouble.cuh"
#include "Color6Component.cuh"

class DirectLightning
{
public:
	__device__ DirectLightning();
	__device__ DirectLightning(const DirectLightning& obj);
	__device__ virtual Color6Component getLightByDistance(double dist) = 0;
	__device__ virtual VectorDouble getPos() = 0;
};