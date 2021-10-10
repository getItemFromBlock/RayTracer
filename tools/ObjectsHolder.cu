#include "ObjectsHolder.cuh"

__host__ __device__ ObjectsHolder::ObjectsHolder() // constructeur - définition
{
	
}

ObjectsHolder::~ObjectsHolder() // Destructeur - définition
{
#ifdef __CUDA_ARCH__ //Determine si la fonction s'execute au niveau GPU
	
#else
	if (hDrawables != (int*)0xcdcdcdcdcdcdcdcd)delete hDrawables;
	if (hLightnings)delete hLightnings;
	if (hAmbientLight)delete hAmbientLight;
	if (hSkyBoxColor)delete hSkyBoxColor;
	if (dTextures) delete[] dTextures;
#endif
}

__host__ __device__ ObjectsHolder & ObjectsHolder::operator=(const ObjectsHolder &)
{
	return *this;
}

ObjectsHolder::ObjectsHolder(const ObjectsHolder & obj)
{
	*this = obj;
}

__host__ int* ObjectsHolder::getDrawables()
{
	return hDrawables;
}

__host__ int* ObjectsHolder::getLightnings()
{
	return hLightnings;
}

__host__ int ObjectsHolder::getDrawablesSize()
{
	return hDsize;
}

__host__ int ObjectsHolder::getLightningsSize()
{
	return hLsize;
}

__global__ void transferTex(ObjectsHolder* gpuObj, int** ptr) {
	gpuObj->syncDTextures(ptr);
}

__host__ void ObjectsHolder::syncTextures(ObjectsHolder* gpuObj, int** ptr) {

}

__device__ void ObjectsHolder::syncDTextures(int** ptr) {
	dTextures = ptr;
}

__device__ void ObjectsHolder::addDDrawable(int* args)
{
	switch (args[0])
	{
	case 0:
	{
		Drawable* obj = new VoidDrawable();
		int* tempsize = new int;
		*tempsize = dDsize;
		dInts->add(tempsize);
		dDrawables->add(obj);
		dDsize = dDrawables->getSize();
		break;
	}
	case 1:
	{
		float x1, x2, x3, rad;
		memcpy(&x1,&args[1],sizeof(float));
		memcpy(&x2,&args[2],sizeof(float));
		memcpy(&x3,&args[3],sizeof(float));
		memcpy(&rad,&args[4],sizeof(float));
		Drawable* obj = new SphereDrawable(Vector3D(x1, x2, x3), rad, Color6Component(args[5], args[6], args[7]), args[13]);
		int* tempsize = new int;
		*tempsize = dDsize;
		dInts->add(tempsize);
		dDrawables->add(obj);
		dDsize = dDrawables->getSize();
		break;
	}
	case 2:
	{
		float a1, a2, a3, b1, b2, b3, c1, c2, c3;
		memcpy(&a1, &args[1], sizeof(float));
		memcpy(&a2, &args[2], sizeof(float));
		memcpy(&a3, &args[3], sizeof(float));
		memcpy(&b1, &args[4], sizeof(float));
		memcpy(&b2, &args[5], sizeof(float));
		memcpy(&b3, &args[6], sizeof(float));
		memcpy(&c1, &args[7], sizeof(float));
		memcpy(&c2, &args[8], sizeof(float));
		memcpy(&c3, &args[9], sizeof(float));
		Drawable* obj = new TriangleDrawable(Vector3D(a1, a2, a3), Vector3D(b1, b2, b3), Vector3D(c1, c2, c3), Color6Component(args[10], args[11], args[12]), args[13]);
		int* tempsize = new int;
		*tempsize = dDsize;
		dInts->add(tempsize);
		dDrawables->add(obj);
		dDsize = dDrawables->getSize();
		break;
	}
	case 3:
	{
		float a1, a2, a3, b1, b2, b3, c1, c2, c3;
		memcpy(&a1, &args[1], sizeof(float));
		memcpy(&a2, &args[2], sizeof(float));
		memcpy(&a3, &args[3], sizeof(float));
		memcpy(&b1, &args[4], sizeof(float));
		memcpy(&b2, &args[5], sizeof(float));
		memcpy(&b3, &args[6], sizeof(float));
		memcpy(&c1, &args[7], sizeof(float));
		memcpy(&c2, &args[8], sizeof(float));
		memcpy(&c3, &args[9], sizeof(float));
		Drawable* obj = new TriangleMirrorDrawable(Vector3D(a1, a2, a3), Vector3D(b1, b2, b3), Vector3D(c1, c2, c3), Color6Component(args[10], args[11], args[12]), args[13]);
		int* tempsize = new int;
		*tempsize = dDsize;
		dInts->add(tempsize);
		dDrawables->add(obj);
		dDsize = dDrawables->getSize();
		break;
	}
	case 4:
	{
		float x1, x2, x3, rad;
		memcpy(&x1, &args[1], sizeof(float));
		memcpy(&x2, &args[2], sizeof(float));
		memcpy(&x3, &args[3], sizeof(float));
		memcpy(&rad, &args[4], sizeof(float));
		Drawable* obj = new SphereMirrorDrawable(Vector3D(x1, x2, x3), rad, Color6Component(args[5], args[6], args[7]), args[13]);
		int* tempsize = new int;
		*tempsize = dDsize;
		dInts->add(tempsize);
		dDrawables->add(obj);
		dDsize = dDrawables->getSize();
		break;
	}
	default:
	{
		Drawable* obj = new VoidDrawable();
		int* tempsize = new int;
		*tempsize = dDsize;
		dInts->add(tempsize);
		dDrawables->add(obj);
		dDsize = dDrawables->getSize();
		break;
	}
	}

	return;
	
}

__device__ void ObjectsHolder::addDLightning(DirectLightning* arg)
{
	dLightnings->add(arg);
	dLsize = dLightnings->getSize();
}

__device__ void ObjectsHolder::modifyDDrawable(int index, int* args)
{
	switch (args[0])
	{
	case 0:
	{
		Drawable* obj = new VoidDrawable();
		int* tempsize = new int;
		*tempsize = index;
		int* u = dInts->at(index);
		Drawable* v = dDrawables->at(index);
		dInts->set(index, tempsize);
		dDrawables->set(index, obj);
		dDsize = dDrawables->getSize();
		delete u;
		delete v;
		break;
	}
	case 1:
	{
		float x1, x2, x3, rad;
		memcpy(&x1, &args[1], sizeof(float));
		memcpy(&x2, &args[2], sizeof(float));
		memcpy(&x3, &args[3], sizeof(float));
		memcpy(&rad, &args[4], sizeof(float));
		Drawable* obj = new SphereDrawable(Vector3D(x1, x2, x3), rad, Color6Component(args[5], args[6], args[7]), args[13]);
		int* tempsize = new int;
		*tempsize = index;
		int* u = dInts->at(index);
		Drawable* v = dDrawables->at(index);
		dInts->set(index, tempsize);
		dDrawables->set(index, obj);
		dDsize = dDrawables->getSize();
		delete u;
		delete v;
		break;
	}
	case 2:
	{
		float a1, a2, a3, b1, b2, b3, c1, c2, c3;
		memcpy(&a1, &args[1], sizeof(float));
		memcpy(&a2, &args[2], sizeof(float));
		memcpy(&a3, &args[3], sizeof(float));
		memcpy(&b1, &args[4], sizeof(float));
		memcpy(&b2, &args[5], sizeof(float));
		memcpy(&b3, &args[6], sizeof(float));
		memcpy(&c1, &args[7], sizeof(float));
		memcpy(&c2, &args[8], sizeof(float));
		memcpy(&c3, &args[9], sizeof(float));
		Drawable* obj = new TriangleDrawable(Vector3D(a1, a2, a3), Vector3D(b1, b2, b3), Vector3D(c1, c2, c3), Color6Component(args[10], args[11], args[12]), args[13]);
		int* tempsize = new int;
		*tempsize = index;
		int* u = dInts->at(index);
		Drawable* v = dDrawables->at(index);
		dInts->set(index, tempsize);
		dDrawables->set(index, obj);
		dDsize = dDrawables->getSize();
		delete u;
		delete v;
		break;
	}
	case 3:
	{
		float a1, a2, a3, b1, b2, b3, c1, c2, c3;
		memcpy(&a1, &args[1], sizeof(float));
		memcpy(&a2, &args[2], sizeof(float));
		memcpy(&a3, &args[3], sizeof(float));
		memcpy(&b1, &args[4], sizeof(float));
		memcpy(&b2, &args[5], sizeof(float));
		memcpy(&b3, &args[6], sizeof(float));
		memcpy(&c1, &args[7], sizeof(float));
		memcpy(&c2, &args[8], sizeof(float));
		memcpy(&c3, &args[9], sizeof(float));
		Drawable* obj = new TriangleMirrorDrawable(Vector3D(a1, a2, a3), Vector3D(b1, b2, b3), Vector3D(c1, c2, c3), Color6Component(args[10], args[11], args[12]), args[13]);
		int* tempsize = new int;
		*tempsize = index;
		int* u = dInts->at(index);
		Drawable* v = dDrawables->at(index);
		dInts->set(index, tempsize);
		dDrawables->set(index, obj);
		dDsize = dDrawables->getSize();
		delete u;
		delete v;
		break;
	}
	case 4:
	{
		float x1, x2, x3, rad;
		memcpy(&x1, &args[1], sizeof(float));
		memcpy(&x2, &args[2], sizeof(float));
		memcpy(&x3, &args[3], sizeof(float));
		memcpy(&rad, &args[4], sizeof(float));
		Drawable* obj = new SphereMirrorDrawable(Vector3D(x1, x2, x3), rad, Color6Component(args[5], args[6], args[7]), args[13]);
		int* tempsize = new int;
		*tempsize = index;
		int* u = dInts->at(index);
		Drawable* v = dDrawables->at(index);
		dInts->set(index, tempsize);
		dDrawables->set(index, obj);
		dDsize = dDrawables->getSize();
		delete u;
		delete v;
		break;
	}
	default:
	{
		Drawable* obj = new VoidDrawable();
		int* tempsize = new int;
		*tempsize = index;
		int* u = dInts->at(index);
		Drawable* v = dDrawables->at(index);
		dInts->set(index, tempsize);
		dDrawables->set(index, obj);
		dDsize = dDrawables->getSize();
		delete u;
		delete v;
		break;
	}
	}

	return;
}

__device__ void ObjectsHolder::modifyDLightning(int index, DirectLightning* arg)
{
	if (index <= dLsize) {
		delete(dLightnings->at(index));
		dLightnings->set(index, arg);
	}
	return;
}


__device__ Drawable* ObjectsHolder::get_d(unsigned int index)
{
	return dDrawables->operator[](index);
}

__device__ DirectLightning * ObjectsHolder::get_l(unsigned int index)
{
	return dLightnings->at(index);
}

__host__ void ObjectsHolder::initFromHostSide() {

	hDsize = 0;
	hLsize = 0;
	hAmbientLight = (int*)malloc(3*sizeof(int));
	hSkyBoxColor = (int*)malloc(3 * sizeof(int));

	dTextures = new int*[32];

}

__device__ void ObjectsHolder::endFromDevice()
{
	for (int n = 0; n < dDsize; n++) {
		Drawable* temp = dDrawables->at(n);
		delete temp;
	}
	for (int n = 0; n < dLsize; n++) {
		DirectLightning* temp = dLightnings->at(n);
		delete temp;
	}

	delete dDrawables;
	delete dLightnings;
	delete dInts;
	if (dAmbientlight) delete dAmbientlight;
}

__device__ void ObjectsHolder::initFromDeviceSide(int* addr) {
	dDsize = 0;
	dLsize = 0;
	dActualSizeX = 2;
	dActualSizeY = 2;
	dFactor = Vector3D();
	dSkyBoxColor = Color6Component();

	dDrawables = new DeviceList<Drawable*>();
	dLightnings = new DeviceList<DirectLightning*>();
	dInts = new DeviceList<int*>();

	dAmbientlight = new AmbientLightning();
	dAmbientlight->setLight(Color6Component());

	dTestList = addr;
}

__global__ void transfertNDrawable(ObjectsHolder* gpuObj, int* arg)
{
	gpuObj->addDDrawable(arg);
}

__device__ Color6Component ObjectsHolder::getLightValueAt(Vector3D* pos, Vector3D* normal)
{
	Color6Component result = dAmbientlight->getLight();
	for (int i = 0; i < dLsize; i++) {
		Vector3D dir = dLightnings->at(i)->getPos().sub(*pos).unitVector();
		Vector3D pos2 = pos->add(dir.mul(-0.0001));
		Ray ray = Ray(pos2, dir);
		bool hit = false;
		for (int j = 0; j < dDsize; j++) {
			if (dDrawables->at(j)->getSubType() != 2) {
				hit = !(dDrawables->at(j)->hit(ray, 0, pos->sub(ray.getDirection()).getLength() - 0.00001).isEmpty);
			}
			if (hit) break;
		}
		if (!hit) {
			float t = dir.unitVector().dot(normal->unitVector());
			if (t >= 0.5) {
				result = result.add(dLightnings->at(i)->getLightByDistance(pos->sub(ray.getOrigin()).getLength()));
			}
			else {
				if (t < 0.0001) t = 0.0;
				Color6Component temp = dLightnings->at(i)->getLightByDistance(pos->sub(ray.getOrigin()).getLength());
				result = result.add(Color6Component(temp.rComponent * 2.0 * t, temp.gComponent * 2.0 * t, temp.bComponent * 2.0 * t));
			}
		}
	}
	return result;
}

__device__ Color6Component ObjectsHolder::hit(Ray * r, float tmin, float tmax, int rmax)
{
	float tm = tmin;
	int reflect = 0;
	bool hit = false;
	bool recalculate;
	float closest = tmax;
	Ray tempRay, actualray;
	actualray = *r;
	tempRay = Ray();
	Vector3D factor = Vector3D(1.0, 1.0, 1.0);

	HitRecord localHit;
	Color6Component localColor;
	Color6Component localLight;

	do {
		recalculate = false;
		for (int i = 0; i < dDsize; i++) {
			HitRecord temp = HitRecord();
			temp = dDrawables->at(i)->hit(actualray, tm, closest);
			//temp = objHit(13*i, actualray, tmin, closest);
			if (!temp.isEmpty) {
				if (dDrawables->at(i)->doReflect()) {
					recalculate = true;
					tempRay = Ray(temp.point, temp.normal);
					tm = -0.0001;
				}
				else {
					hit = true;
					recalculate = false;
				}
				closest = temp.t;
				localHit = temp;
				//Color6Component tempC = dDrawables->at(i)->getColor(&temp);
				int u = dTestList[13 * i];
				Color6Component tempC = (u==4||u==1)?Color6Component(dTestList[13*i+5], dTestList[13*i+6], dTestList[13*i+7]):Color6Component(dTestList[13*i+10], dTestList[13*i+11], dTestList[13*i+12]);
				localColor = Color6Component(tempC.rComponent*factor.getX(), tempC.gComponent*factor.getY(), tempC.bComponent*factor.getZ());
				localLight = getLightValueAt(&temp.point,&temp.normal);
			}
		}
		if (recalculate) {
			factor.setX(localColor.rComponent / 32767.0);
			factor.setY(localColor.gComponent / 32767.0);
			factor.setZ(localColor.bComponent / 32767.0);
			actualray = tempRay;
			closest = tmax - closest;
			reflect++;
			hit = false;
		}
	} while (recalculate && reflect < rmax);
	if (!hit) {
		float t = 0.5 * (actualray.getDirection().unitVector().getY() + 1.0);
		localColor = Color6Component(((1.0 - t) + t * dSkyBoxColor.rComponent), ((1.0 - t) + t * dSkyBoxColor.gComponent), ((1.0 - t) + t * dSkyBoxColor.bComponent));
	}
	else {
		localColor = localColor.add(localLight);
	}
	return localColor;
}

__global__ void transfertNLightnings(ObjectsHolder* gpuObj, int* arg)
{
	switch (arg[0])
	{
	case 0:
	{
		VoidDirectLightning* obj = new VoidDirectLightning();
		gpuObj->addDLightning(obj);
		break;
	}
	case 1:
	{
		float x1, x2, x3, att;
		memcpy(&x1,&arg[1],sizeof(float));
		memcpy(&x2,&arg[2],sizeof(float));
		memcpy(&x3,&arg[3],sizeof(float));
		memcpy(&att,&arg[7],sizeof(float));
		PointDirectLightning* obj = new PointDirectLightning(Vector3D(x1, x2, x3), Color6Component(arg[4], arg[5], arg[6]), att);
		gpuObj->addDLightning(obj);
		break;
	}
	default:
	{
		VoidDirectLightning* obj = new VoidDirectLightning();
		gpuObj->addDLightning(obj);
		break;
	}
	}

	return;
}

__global__ void transfertNTexture(ObjectsHolder* gpuObj, unsigned char* arg)
{
	
	return;
}

__global__ void transfertEDrawable(ObjectsHolder* gpuObj, int* arg, int index)
{
	gpuObj->modifyDDrawable(index, arg);
	return;
}

__global__ void transfertELightnings(ObjectsHolder* gpuObj, int* arg, int index)
{
	switch (arg[0])
	{
	case 0:
	{
		VoidDirectLightning* obj = new VoidDirectLightning();
		gpuObj->modifyDLightning(index, obj);
		break;
	}
	case 1:
	{
		float x1, x2, x3, att;
		memcpy(&x1, &arg[1], sizeof(float));
		memcpy(&x2, &arg[2], sizeof(float));
		memcpy(&x3, &arg[3], sizeof(float));
		memcpy(&att, &arg[7], sizeof(float));
		PointDirectLightning* obj = new PointDirectLightning(Vector3D(x1, x2, x3), Color6Component(arg[4], arg[5], arg[6]), att);
		gpuObj->modifyDLightning(index, obj);
		break;
	}
	default:
	{
		VoidDirectLightning* obj = new VoidDirectLightning();
		gpuObj->modifyDLightning(index, obj);
		break;
	}
	}

	return;
}

__global__ void transfertSkyBoxColor(ObjectsHolder* gpuObj, int* arg)
{
	Color6Component obj = Color6Component(arg[0], arg[1], arg[2]);
	gpuObj->dSkyBoxColor = obj;
	return;
}

__global__ void transfertAmbientLight(ObjectsHolder* gpuObj, int* arg)
{
	AmbientLightning* obj = new AmbientLightning(Color6Component(arg[0], arg[1], arg[2]));
	if (gpuObj->dAmbientlight) {
		delete gpuObj->dAmbientlight;
	}
	gpuObj->dAmbientlight = obj;
	return;
}

__global__ void transfertFactor(ObjectsHolder* gpuObj, int* arg)
{
	float x, y, z;
	memcpy(&x,&arg[0],sizeof(float));
	memcpy(&y,&arg[1],sizeof(float));
	memcpy(&z,&arg[2],sizeof(float));
	Vector3D obj = Vector3D(x, y, z);
	gpuObj->dFactor = obj;
	return;
}

__host__ int ObjectsHolder::addDrawable(ObjectsHolder* gpuObj, int * arg)
{
	if (hDsize == 0) {
		hDrawables = (int*)malloc(objectSizeA * sizeof(int));
	}
	else {
		hDrawables = (int*)realloc(hDrawables, (hDsize+1) * objectSizeA * sizeof(int));
	}
	hDsize++;
	for (int j = 0; j < objectSizeA; j++) {
		hDrawables[(hDsize - 1)*objectSizeA + j] = arg[j];
	}
	int* gpuArgs;
	gpuErrchk(cudaMalloc((void**)&gpuArgs, objectSizeA * sizeof(int)));
	gpuErrchk(cudaMemcpy(gpuArgs, arg, objectSizeA * sizeof(int), cudaMemcpyHostToDevice));
	transfertNDrawable <<<1, 1>>> (gpuObj, gpuArgs);
	
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaFree(gpuArgs);
	return hDsize-1;
}

__host__ int ObjectsHolder::addLightning(ObjectsHolder* gpuObj, int * arg)
{
	if (hLsize == 0) {
		hLightnings = (int*)malloc(objectSizeB * sizeof(int));
	}
	else {
		hLightnings = (int*)realloc(hLightnings, (hLsize + 1) * objectSizeB * sizeof(int));
	}
	hLsize++;
	for (int j = 0; j < objectSizeB; j++) {
		hLightnings[(hLsize-1)*objectSizeB + j] = arg[j];
	}

	int* gpuArgs;
	gpuErrchk(cudaMalloc((void**)&gpuArgs, objectSizeB * sizeof(int)));
	gpuErrchk(cudaMemcpy(gpuArgs, arg, objectSizeB * sizeof(int), cudaMemcpyHostToDevice));
	transfertNLightnings <<<1, 1 >>> (gpuObj, gpuArgs);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaFree(gpuArgs);
	return hLsize-1;
}

__host__ int ObjectsHolder::addTexture(ObjectsHolder* gpuObj, unsigned char* tex)
{
	hTsize++;
	if (!tex) return -1;
	int sX = (int)(tex[0]) * 256 + (int)(tex[1]);
	int sY = (int)(tex[2]) * 256 + (int)(tex[3]);
	unsigned char* gpuArgs;
	gpuErrchk(cudaMalloc((void**)&gpuArgs, (sX*sY*3+4)*sizeof(unsigned char)));
	gpuErrchk(cudaMemcpy(gpuArgs, tex, (sX * sY * 3 + 4) * sizeof(unsigned char), cudaMemcpyHostToDevice));
	transfertNTexture << <1, 1 >> > (gpuObj, gpuArgs);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	return hTsize - 1;
}

__host__ int ObjectsHolder::modifyDrawable(ObjectsHolder* gpuObj, int index, int * arg)
{
	if (index > hDsize) {
		return -1;
	}
	for (int j = 0; j < objectSizeA; j++) {
		hDrawables[index*objectSizeA + j] = arg[j];
	}
	int* gpuArgs;
	gpuErrchk(cudaMalloc((void**)&gpuArgs, objectSizeA * sizeof(int)));
	gpuErrchk(cudaMemcpy(gpuArgs, arg, objectSizeA * sizeof(int), cudaMemcpyHostToDevice));
	transfertEDrawable <<<1, 1 >>> (gpuObj, gpuArgs, index);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaFree(gpuArgs);
	return index;
}

__host__ int ObjectsHolder::modifyLightning(ObjectsHolder* gpuObj, int index, int * arg)
{
	if (index > hLsize) {
		return -1;
	}
	for (int j = 0; j < objectSizeB; j++) {
		hLightnings[index*objectSizeB + j] = arg[j];
	}
	int* gpuArgs;
	gpuErrchk(cudaMalloc((void**)&gpuArgs, objectSizeB * sizeof(int)));
	gpuErrchk(cudaMemcpy(gpuArgs, arg, objectSizeB * sizeof(int), cudaMemcpyHostToDevice));
	transfertELightnings << <1, 1 >> > (gpuObj, gpuArgs, index);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaFree(gpuArgs);
	return index;
}

__host__ void ObjectsHolder::setSkyBoxColor(ObjectsHolder * gpuObj, int* arg)
{
	if (hSkyBoxColor) {
		delete hSkyBoxColor;
	}
	hSkyBoxColor = (int*)malloc(3 * sizeof(int));
	hSkyBoxColor[0] = arg[0];
	hSkyBoxColor[1] = arg[1];
	hSkyBoxColor[2] = arg[2];

	int* gpuArgs;
	gpuErrchk(cudaMalloc((void**)&gpuArgs, 3 * sizeof(int)));
	gpuErrchk(cudaMemcpy(gpuArgs, arg, 3 * sizeof(int), cudaMemcpyHostToDevice));
	transfertSkyBoxColor <<<1, 1 >>> (gpuObj, gpuArgs);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaFree(gpuArgs);
	return;
}

__host__ void ObjectsHolder::setAmbientLight(ObjectsHolder * gpuObj, int* arg)
{
	if (hAmbientLight) {
		delete hAmbientLight;
	}
	hAmbientLight = (int*)malloc(3 * sizeof(int));
	hAmbientLight[0] = arg[0];
	hAmbientLight[1] = arg[1];
	hAmbientLight[2] = arg[2];

	int* gpuArgs;
	gpuErrchk(cudaMalloc((void**)&gpuArgs, 3 * sizeof(int)));
	gpuErrchk(cudaMemcpy(gpuArgs, arg, 3 * sizeof(int), cudaMemcpyHostToDevice));
	transfertAmbientLight <<<1, 1 >>> (gpuObj, gpuArgs);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaFree(gpuArgs);
	return;
}

__host__ void ObjectsHolder::setFactor(ObjectsHolder * gpuObj, Vector3D arg)
{
	int* tempArg = (int*)malloc(3 * sizeof(int));
	float d0 = arg.getX();
	float d1 = arg.getY();
	float d2 = arg.getZ();
	memcpy(&tempArg[0], &d0,sizeof(int));
	memcpy(&tempArg[1], &d1, sizeof(int));
	memcpy(&tempArg[2], &d2, sizeof(int));

	int* gpuArgs;
	gpuErrchk(cudaMalloc((void**)&gpuArgs, 3 * sizeof(int)));
	gpuErrchk(cudaMemcpy(gpuArgs, tempArg, 3 * sizeof(int), cudaMemcpyHostToDevice));
	transfertFactor <<<1, 1 >>> (gpuObj, gpuArgs);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaFree(gpuArgs);
	free(tempArg);
	return;
}