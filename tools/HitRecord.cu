#include "HitRecord.cuh"

__host__ __device__ HitRecord::HitRecord(Vector3D pointT, Vector3D normalT, float T) : point(pointT), normal(normalT), t(T) // constructeur - définition
{
	isEmpty = false;
}

__host__ __device__ HitRecord::HitRecord() : point(Vector3D()), normal(Vector3D()), t(100000), isEmpty(true)
{
	isEmpty = true;
}

__host__ __device__ HitRecord::~HitRecord() // Destructeur - définition
{

}

__host__ __device__ void HitRecord::setFaceNormal(Ray r, Vector3D outward_normal)
{
	front_face = r.getDirection().dot(outward_normal) < 0;
	normal = front_face ? outward_normal : outward_normal.mul(-1.0);
}