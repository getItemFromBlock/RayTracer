#pragma once
#include "../tools/HitRecord.cuh"
#include "Drawable.cuh"
class SphereDrawable :public Drawable
{
public:
	__device__ SphereDrawable(Vector3D, float, Color6Component, int);
	__device__ SphereDrawable();
	__device__ SphereDrawable(const Drawable& obj);
	__device__ ~SphereDrawable();

	__device__ virtual HitRecord Drawable::hit(Ray r, float tmin, float tmax) override;
	__device__ virtual Color6Component Drawable::getColor(HitRecord* hit) override;
	__device__ virtual bool Drawable::doReflect() override;
	__device__ virtual int Drawable::getSubType() override;
private:
	Vector3D pos;
	const float radius;
	const Color6Component color;
	const int type;
};