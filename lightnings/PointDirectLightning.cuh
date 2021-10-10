#pragma once
#include "DirectLightning.cuh"
#include "../tools/Color6Component.cuh"

class PointDirectLightning :public DirectLightning
{
public:
	__device__ PointDirectLightning();
	__device__ PointDirectLightning(Vector3D origin, Color6Component lightcolor, float atenuationByDist);
	__device__ PointDirectLightning(const DirectLightning& obj);
	__device__ ~PointDirectLightning();

	__device__ virtual Color6Component getLightByDistance(float dist) override;
	__device__ virtual Vector3D getPos() override;
private:
	Vector3D center;
	Color6Component light;
	float atenuation;
};