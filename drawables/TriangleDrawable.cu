#include "TriangleDrawable.cuh"


__device__ TriangleDrawable::TriangleDrawable(Vector3D point1, Vector3D point2, Vector3D point3, Color6Component couleur, int subType):
	A(point1), B(point2), C(point3), color(couleur), type(subType) // constructeur - définition
{
	AB = B.sub(A);
	AC = C.sub(A);
	normal = AB.cross(AC);
}

__device__ TriangleDrawable::TriangleDrawable():
	A(Vector3D()), B(Vector3D()), C(Vector3D()), color(Color6Component()), type(0)
{
	normal = Vector3D(0, 1, 0);
	AB = Vector3D();
	AC = Vector3D();
}

__device__ TriangleDrawable::TriangleDrawable(const Drawable& obj) : Drawable(obj), type(0)
{

}

__device__ TriangleDrawable::~TriangleDrawable() // Destructeur - définition
{

}

__device__ HitRecord TriangleDrawable::hit(Ray r, float tmin, float tmax)
{
	HitRecord rec = HitRecord();
	float det = -r.getDirection().dot(normal);
	float invdet = 1.0 / det;
	Vector3D AO = r.getOrigin().sub(A);
	Vector3D DAO = AO.cross(r.getDirection());
	rec.u = AC.dot(DAO)*invdet;
	rec.v = -AB.dot(DAO)*invdet;
	float t = AO.dot(normal)*invdet;
	if (det >= 1e-6 && t >= tmin && t <= tmax && rec.u >= 0.0 && rec.v >= 0.0 && rec.u + rec.v <= 1.0) {
		rec.normal = normal;
		rec.point = r.getOrigin().add(r.getDirection().mul(t));
		rec.t = t;
		rec.front_face = true;
		rec.isEmpty = false;
	}
	return rec;
}

__device__ Color6Component TriangleDrawable::getColor(HitRecord* hit)
{
	return color;
}

__device__ bool TriangleDrawable::doReflect()
{
	return false;
}

__device__ int TriangleDrawable::getSubType() {
	return type;
}