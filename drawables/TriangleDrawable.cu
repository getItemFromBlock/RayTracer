#include "TriangleDrawable.cuh"


__device__ TriangleDrawable::TriangleDrawable(VectorDouble point1, VectorDouble point2, VectorDouble point3, Color6Component couleur):
	A(point1), B(point2), C(point3), color(couleur) // constructeur - définition
{
	AB = B.sub(A);
	AC = C.sub(A);
	normal = AB.cross(AC);
}

__device__ TriangleDrawable::TriangleDrawable():
	A(VectorDouble()), B(VectorDouble()), C(VectorDouble()), color(Color6Component())
{
	normal = VectorDouble(0, 1, 0);
	AB = VectorDouble();
	AC = VectorDouble();
}

__device__ TriangleDrawable::TriangleDrawable(const Drawable& obj) : Drawable(obj)
{

}

__device__ TriangleDrawable::~TriangleDrawable() // Destructeur - définition
{

}

__device__ HitRecord TriangleDrawable::hit(Ray r, double tmin, double tmax)
{
	HitRecord rec = HitRecord();
	double det = -r.getDirection().dot(normal);
	double invdet = 1.0 / det;
	VectorDouble AO = r.getOrigin().sub(A);
	VectorDouble DAO = AO.cross(r.getDirection());
	rec.u = AC.dot(DAO)*invdet;
	rec.v = -AB.dot(DAO)*invdet;
	double t = AO.dot(normal)*invdet;
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

__device__ double TriangleDrawable::getVHitBox(VectorDouble* position)
{
	Ray down = Ray(VectorDouble(position->getX(), position->getY(), position->getZ()), VectorDouble(0, -1, 0));
	double det = -down.getDirection().dot(normal);
	double invdet = 1.0 / det;
	VectorDouble AO = down.getOrigin().sub(A);
	VectorDouble DAO = AO.cross(down.getDirection());
	double ub = AC.dot(DAO)*invdet;
	double vb = -AB.dot(DAO)*invdet;
	double t = AO.dot(normal)*invdet;

	if (det >= 1e-6 && t >= 0 && t <= 0.6 && ub >= -0.2 && vb >= -0.2 && ub + vb <= 1.2) {
		return down.getOrigin().add(down.getDirection().mul(t)).getY();
	}
	return -50000;
}