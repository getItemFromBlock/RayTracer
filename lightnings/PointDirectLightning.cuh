#pragma once
#include "DirectLightning.cuh"
#include "Color6Component.cuh"

class PointDirectLightning :public DirectLightning
{
public:
	__device__ PointDirectLightning();
	__device__ PointDirectLightning(VectorDouble origin, Color6Component lightcolor, double atenuationByDist);
	__device__ PointDirectLightning(const DirectLightning& obj);
	__device__ ~PointDirectLightning();

	__device__ virtual Color6Component getLightByDistance(double dist) override;
	__device__ virtual VectorDouble getPos() override;
private:
	VectorDouble center;
	Color6Component light;
	double atenuation;
};