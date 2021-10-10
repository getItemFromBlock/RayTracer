#pragma once
#include "../tools/HitRecord.cuh"
#include "Drawable.cuh"
class SphereMirrorDrawable :public Drawable
{
public:
	__device__ SphereMirrorDrawable(Vector3D, float, Color6Component, int);
	__device__ SphereMirrorDrawable();
	__device__ SphereMirrorDrawable(const Drawable& obj);
	__device__ ~SphereMirrorDrawable();

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