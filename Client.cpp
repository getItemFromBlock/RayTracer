#include <windows.h>
#include "WindowsHolder.h"
#include "VectorDouble.cuh"
#include <cmath>
#include <corecrt_math_defines.h>

extern void setResolution(int x, int y);
extern void initRayTracer();
extern int* renderImage();
extern void setCameraArgs(VectorDouble arg1, VectorDouble arg2, VectorDouble arg3, double arg4);
extern void endRayTracer();
extern int addDrawableObj(int* arg);
extern int addLightningObj(int* arg);
extern void setSkyBoxColorRGB(int* arg);
extern void setInitialised();

static WindowsHolder* windowC;

static VectorDouble targetBase = VectorDouble(1, -0.5, 0);
static VectorDouble camBase = VectorDouble(2, 0, 0);

static int wWidth = 100;
static int wHeight = 100;

static constexpr double speedBase = 0.07;
static constexpr double speedCrouch = 0.025;

static double rotX = 0;
static double rotY = 0;

static bool refreshScreenSize() {
	int maxX = windowC->getDrawSizeX();
	int maxY = windowC->getDrawSizeY();

	if (wWidth != maxX || wHeight != maxY) {
		if (maxX != 0 && maxY != 0) {
			wWidth = maxX;
			wHeight = maxY;
			setResolution(wWidth, wHeight);
			return true;
		}
	}
	return false;

}

static double toRadians(double angle) {
	return angle/180*M_PI;
}

static double cut(double value, double min, double max) {
	if (value < min) return min;
	if (value > max) return max;
	return value;
}

static void refreshRotation() {
	setCameraArgs(camBase, VectorDouble(cos(toRadians(rotX))*cos(toRadians(rotY)), sin(toRadians(rotY)), sin(toRadians(rotX))*cos(toRadians(rotY))), VectorDouble(0, 1, 0), 90);
}

static void refreshScreenValue() {
	int* screen = renderImage();
	windowC->setOutputScreen(screen);
}

static VectorDouble getPlayerDir() {
	VectorDouble result = VectorDouble(0, 0, 0);
	const double speed = windowC->getInputValue(VK_CONTROL)?speedCrouch:speedBase;
	if (windowC->getInputValue('Z')) {
		result.setX(1.0);
	}
	if (windowC->getInputValue('S')) {
		result.setX(result.getX() - 1.0);
	}
	if (windowC->getInputValue('D')) {
		result.setZ(1.0);
	}
	if (windowC->getInputValue('Q')) {
		result.setZ(result.getZ() - 1.0);
	}
	if (result.length_squared() > 0.001) {
		result = result.unitVector().mul(speed);
	}
	return result;
}

static int addSphereD(float posX, float posY, float posZ, float rad, int colR, int colG, int colB, bool mirror) {
	int* obj = (int*)malloc(13 * sizeof(int));
	obj[0] = mirror?4:1;
	obj[1] = *(int*)&(posX);
	obj[2] = *(int*)&(posY);
	obj[3] = *(int*)&(posZ);
	obj[4] = *(int*)&(rad);
	obj[5] = colR;
	obj[6] = colG;
	obj[7] = colB;
	int t = addDrawableObj(obj);
	free(obj);
	return t;
}

static int addTriangleD(float a1, float a2, float a3, float b1, float b2, float b3, float c1, float c2, float c3, int colR, int colG, int colB, bool mirror) {
	int* obj = (int*)malloc(13 * sizeof(int));
	obj[0] = mirror?3:2;
	obj[1] = *(int*)&(a1);
	obj[2] = *(int*)&(a2);
	obj[3] = *(int*)&(a3);
	obj[4] = *(int*)&(b1);
	obj[5] = *(int*)&(b2);
	obj[6] = *(int*)&(b3);
	obj[7] = *(int*)&(c1);
	obj[8] = *(int*)&(c2);
	obj[9] = *(int*)&(c3);
	obj[10] = colR;
	obj[11] = colG;
	obj[12] = colB;
	int t = addDrawableObj(obj);
	free(obj);
	return t;
}

static void initClient() {
	initRayTracer();
	setCameraArgs(VectorDouble(2, 0, 0), VectorDouble(1, -0.5, 0), VectorDouble(0, 1, 0), 90);
	
	addSphereD(0, 0, 1, 0.7, 0, 16000, 0, false);
	addSphereD(0, 0.2, -1, 0.3, 8000, 0, 16000, false);

	addTriangleD(-50, -1, -50, -50, -1, 50, 50, -1, 50, 8000, 1000, 1000, false);
	addTriangleD(50, -1, 50, 50, -1, -50, -50, -1, -50, 8000, 2000, 1000, false);
	addTriangleD(-1, -0.5, -1, -1, -1, 1, 1, -1, 0, 27195, 27195, 29490, true);

	int* obj3 = (int*)malloc(8 * sizeof(int));
	obj3[0] = 1;
	float posX2 = 1;
	float posY2 = 10;
	float posZ2 = 5;
	float att = 99;
	obj3[1] = *(int*)&(posX2);
	obj3[2] = *(int*)&(posY2);
	obj3[3] = *(int*)&(posZ2);
	obj3[7] = *(int*)&(att);
	obj3[4] = 8000;
	obj3[5] = 8000;
	obj3[6] = 8000;
	addLightningObj(obj3);
	free(obj3);

	int* obj4 = (int*)malloc(3 * sizeof(int));
	obj4[0] = 22937;
	obj4[1] = 26214;
	obj4[2] = 32767;
	setSkyBoxColorRGB(obj4);
	free(obj4);
	setInitialised();
}

static DWORD WINAPI clientMain(LPVOID lpParameter) {
	windowC = ((WindowsHolder*)lpParameter);

	initClient();

	while (windowC->isCaptured() || !windowC->getInputValue(VK_CONTROL)) {
		refreshScreenSize();

		if (windowC->isCaptured()) {
			int test = windowC->getMousePosX();
			rotX = windowC->getMousePosX() / 5.0 + rotX;
			while (rotX < 0) rotX += 360;
			while (rotX >= 360) rotX -= 360;
			rotY = windowC->getMousePosY() / 5.0 + rotY;
			rotY = cut(rotY, -90, 90);
			VectorDouble playerDir = getPlayerDir();
			camBase.setX(camBase.getX() + cos(toRadians(rotX))*(-playerDir.getX()) + sin(toRadians(rotX))*playerDir.getZ());
			camBase.setZ(camBase.getZ() + sin(toRadians(rotX))*(-playerDir.getX()) - cos(toRadians(rotX))*playerDir.getZ());
			refreshRotation();
			windowC->setMousePosInput(0, 0);
		}
		

		if (windowC->getInputValue(VK_ESCAPE)) {
			windowC->setInputValue(VK_ESCAPE, false);
			windowC->setCaptured(!windowC->isCaptured());
		}

		refreshScreenValue();

		InvalidateRect(windowC->windowH, NULL, TRUE);
		UpdateWindow(windowC->windowH);
		windowC->sizing = false;

	}
	endRayTracer();
	windowC->exitWindow = true;

	return 0;
}