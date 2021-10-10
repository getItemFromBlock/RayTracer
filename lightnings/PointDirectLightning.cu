#include "PointDirectLightning.cuh"

__device__ PointDirectLightning::PointDirectLightning()
{
	center = Vector3D();
	light = Color6Component();
	atenuation = 100.0;
}

__device__ PointDirectLightning::PointDirectLightning(Vector3D origin, Color6Component lightcolor, float atenuationByDist)
{
	center = origin;
	light = lightcolor;
	atenuation = atenuationByDist;
}

__device__ PointDirectLightning::PointDirectLightning(const DirectLightning& obj) : DirectLightning(obj)
{
	atenuation = 0;
}

__device__ PointDirectLightning::~PointDirectLightning()
{
}

__device__ Vector3D PointDirectLightning::getPos()
{
	return center;
}

__device__ Color6Component PointDirectLightning::getLightByDistance(float dist)
{
	float r = light.rComponent*powf(atenuation / 100.0, dist);
	float g = light.gComponent*powf(atenuation / 100.0, dist);
	float b = light.bComponent*powf(atenuation / 100.0, dist);
	return Color6Component(r, g, b);
}