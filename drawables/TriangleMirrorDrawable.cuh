#pragma once
#include "HitRecord.cuh"
#include "Drawable.cuh"
class TriangleMirrorDrawable :public Drawable
{
public:
	__device__ TriangleMirrorDrawable(VectorDouble, VectorDouble, VectorDouble, Color6Component);
	__device__ TriangleMirrorDrawable();
	__device__ TriangleMirrorDrawable(const Drawable& obj);
	__device__ ~TriangleMirrorDrawable();

	__device__ virtual HitRecord Drawable::hit(Ray r, double tmin, double tmax) override;
	__device__ virtual Color6Component Drawable::getColor(HitRecord* hit) override;
	__device__ virtual double Drawable::getVHitBox(VectorDouble* position) override;
	__device__ virtual bool Drawable::doReflect() override;
private:
	VectorDouble A, B, C, normal, AB, AC;
	const Color6Component color;
};