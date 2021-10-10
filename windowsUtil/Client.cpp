#include <windows.h>
#include "WindowsHolder.h"
#include "../tools/Vector3D.cuh"
#include "../tools/ObjectsHolder.cuh"
#include <cmath>
#include <chrono>
#include <corecrt_math_defines.h>
#include "../tools/MathHelper.h"
#include "../tools/GuiHelper.h"
#include "../tools/ThreadArgs.h"

static long long int timeSinceEpochMillisec() {
	using namespace std::chrono;
	return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

extern void setResolution(int x, int y);
extern void initRayTracer(ObjectsHolder* obj);
extern int* renderImage();
extern void setCameraArgs(Vector3D arg1, Vector3D arg2, Vector3D arg3, float arg4);
extern void endRayTracer();
extern int addDrawableObj(int* arg);
extern int addLightningObj(int* arg);
extern void setSkyBoxColorRGB(int* arg);
extern void setInitialised();
extern int modDrawableObj(int index, int* arg);

static WindowsHolder* windowC;
static ThreadArgs* clientArgs;
static ObjectsHolder* clientHolder;

static float baseHeight = 1.0;
static float crouchHeight = 0.6;

static int wWidth = 100;
static int wHeight = 100;

static bool tr = false;
static bool textr = false;

static int sp1 = 0, player_head = 0, player_pos = 0;

static unsigned char** textures;

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

static float toRadians(float angle) {
	return angle / 180 * M_PI;
}

static float cut(float value, float min, float max) {
	if (value < min) return min;
	if (value > max) return max;
	return value;
}

static void refreshRotation(Vector3D pos, float vfov) {
	setCameraArgs(pos, Vector3D(cos(toRadians(windowC->rotX)) * cos(toRadians(windowC->rotY)), sin(toRadians(windowC->rotY)), sin(toRadians(windowC->rotX)) * cos(toRadians(windowC->rotY))), Vector3D(0, 1, 0), vfov);
}

static void refreshScreenValue(int frameTime, bool menu) {
	int* screen = renderImage();
	renderGUI(screen, frameTime, menu);
	drawString(screen, "PosX:", 10, 25, 0x00000000);
	drawString(screen, "PosY:", 10, 40, 0x00000000);
	drawString(screen, "PosZ:", 10, 55, 0x00000000);
	char* x = intToChar(windowC->playerPos.getX() * 1000);
	char* y = intToChar(windowC->playerPos.getY() * 1000);
	char* z = intToChar(windowC->playerPos.getZ() * 1000);
	drawString(screen, x, 48, 25, 0x00008000);
	drawString(screen, y, 48, 40, 0x00008000);
	drawString(screen, z, 48, 55, 0x00008000);
	delete[] x;
	delete[] y;
	delete[] z;
	//drawTex(screen,textures[textr?2:1],10,100);
	windowC->setOutputScreen(screen);
}

static int addSphereD(float posX, float posY, float posZ, float rad, int colR, int colG, int colB, bool mirror, int subData) {
	int* obj = new int[14];
	obj[0] = mirror ? 4 : 1;
	obj[1] = *(int*)&(posX);
	obj[2] = *(int*)&(posY);
	obj[3] = *(int*)&(posZ);
	obj[4] = *(int*)&(rad);
	obj[5] = colR;
	obj[6] = colG;
	obj[7] = colB;
	obj[13] = subData;
	int t = addDrawableObj(obj);
	delete[] obj;
	return t;
}

static int addTriangleD(float a1, float a2, float a3, float b1, float b2, float b3, float c1, float c2, float c3, int colR, int colG, int colB, bool mirror) {
	int* obj = new int[14];
	obj[0] = mirror ? 3 : 2;
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
	obj[13] = 1;
	int t = addDrawableObj(obj);
	delete[] obj;
	return t;
}

static int addPointL(float x, float y, float z, float att, int colR, int colG, int colB) {
	int* obj = new int[8];
	obj[0] = 1;
	memcpy(&obj[1], &x, sizeof(int));
	memcpy(&obj[2], &y, sizeof(int));
	memcpy(&obj[3], &z, sizeof(int));
	memcpy(&obj[7], &att, sizeof(int));
	obj[4] = colR;
	obj[5] = colG;
	obj[6] = colB;
	int t = addLightningObj(obj);
	delete[] obj;
	return t;
}

static int modSphereD(int index, float posX, float posY, float posZ, float rad, int colR, int colG, int colB, bool mirror, bool collide) {
	int* obj = new int[14];
	obj[0] = mirror ? 4 : 1;
	memcpy(&obj[1], &posX, sizeof(int));
	memcpy(&obj[2], &posY, sizeof(int));
	memcpy(&obj[3], &posZ, sizeof(int));
	memcpy(&obj[4], &rad, sizeof(int));
	obj[5] = colR;
	obj[6] = colG;
	obj[7] = colB;
	obj[13] = collide ? 0 : 1;
	int t = modDrawableObj(index, obj);
	delete[] obj;
	return t;
}

static void initClient() {
	initRayTracer(clientHolder);
	setCameraArgs(Vector3D(2, 0, 0), Vector3D(1, -0.5, 0), Vector3D(0, 1, 0), 90);

	addSphereD(0, 0, 1, 0.7, 0, 16000, 0, false, 0);
	sp1 = addSphereD(0, 0.2, -1, 0.3, 8000, 0, 16000, false, 0);

	player_head = addSphereD(2, 0, 0, 0.3, 20000, 11000, 2100, false, 1);
	player_pos = addSphereD(2, -1, 0, 0.3, 3000, 3000, 20000, false, 1);

	addSphereD(1, 20, 5, 1, 50000, 50000, 20000, false, 2);

	addSphereD(1.5, 0.2, 1, 0.6, 29490, 27195, 27195, true, 0);
	//addSphereD(-7, 0, 1, -0.6, 29490, 27195, 27195, true, 0);
	//addSphereD(-10, 0, 1, -0.6, 29490, 27195, 27195, true, 2);

	addTriangleD(-50, -1, -50, -50, -1, 50, 50, -1, 50, 8000, 1000, 1000, false);
	addTriangleD(50, -1, 50, 50, -1, -50, -50, -1, -50, 8000, 2000, 1000, false);
	addTriangleD(1, 0.51, -1, 1, -0.99, 1, 3, -0.99, 0, 27195, 27195, 29490, true);
	addTriangleD(1, -1, 1, 1, 0.5, -1, 3, -1, 0, 9195, 7195, 9490, false);

	addPointL(1.0, 20.0, 5.0, 100.0, 8000, 8000, 8000);

	int* obj4 = new int[3];
	obj4[0] = 22937;
	obj4[1] = 26214;
	obj4[2] = 32767;
	setSkyBoxColorRGB(obj4);
	delete[] obj4;
	setInitialised();
}

static void clearClient() {
	endRayTracer();
	endChars();
	delete[] textures[0];
	delete[] textures[1];
	delete[] textures[2];
	delete[] textures;
}

static DWORD WINAPI clientMain(LPVOID lpParameter) {
	clientArgs = ((ThreadArgs*)lpParameter);
	windowC = clientArgs->getWHolder();
	clientHolder = clientArgs->getOHolder();

	initClient();
	int zergt = initChars();
	if (zergt == 1) {
		MessageBox(NULL, "Couldn't open file.", NULL,
			MB_OK | MB_ICONEXCLAMATION);
		PostMessage(NULL, WM_CLOSE, 0, 0L);
	}
	if (zergt == 2) {
		MessageBox(NULL, "Wrong file.", NULL,
			MB_OK | MB_ICONEXCLAMATION);
		PostMessage(NULL, WM_CLOSE, 0, 0L);
	}
	if (zergt == 128) {
		MessageBox(NULL, "Test", NULL,
			MB_OK | MB_ICONEXCLAMATION);
		PostMessage(NULL, WM_CLOSE, 0, 0L);
	}
	textures = new unsigned char* [3];
	int qserfd = loadTexture(&(textures[0]), "./pictures/pingas.ppm");
	int qserft = loadTexture(&(textures[1]), "./pictures/bananas.ppm");
	int qserfr = loadTexture(&(textures[2]), "./pictures/mario.ppm");
	if (qserfd != 0) {
		char* rslt = intToChar(qserfd);
		MessageBox(NULL, rslt, NULL,
			MB_OK | MB_ICONEXCLAMATION);
		PostMessage(NULL, WM_CLOSE, 0, 0L);
		delete[] rslt;
	}
	if (qserft != 0) {
		char* rslt = intToChar(qserfd);
		MessageBox(NULL, rslt, NULL,
			MB_OK | MB_ICONEXCLAMATION);
		PostMessage(NULL, WM_CLOSE, 0, 0L);
		delete[] rslt;
	}
	if (qserfr != 0) {
		char* rslt = intToChar(qserfd);
		MessageBox(NULL, rslt, NULL,
			MB_OK | MB_ICONEXCLAMATION);
		PostMessage(NULL, WM_CLOSE, 0, 0L);
		delete[] rslt;
	}
	int frameTime = 0;
	long long int dt = timeSinceEpochMillisec();
	long long int dt2 = 0;
	Vector3D tempPos;
	while (true) {
		refreshScreenSize();

		if (windowC->isCaptured()) {
			int test = windowC->getMousePosX();
			float rX = windowC->rotX, rY = windowC->rotY;
			rX = windowC->getMousePosX() / 5.0 + rX;
			while (rX < 0) rX += 360;
			while (rX >= 360) rX -= 360;
			rY = windowC->getMousePosY() / 5.0 + rY;
			rY = cut(rY, -89.9, 89.9);
			windowC->rotX = rX;
			windowC->rotY = rY;
			tempPos = windowC->playerPos;
			refreshRotation(tempPos.add(Vector3D(0, (windowC->getInputValue(VK_CONTROL) ? crouchHeight : baseHeight), 0)), windowC->getInputValue('W') ? 5 : 90);
			if (!tr && windowC->getInputValue(VK_CONTROL)) {
				tr = true;
				sp1 = modSphereD(sp1, 0, (float)(prng()), -1, 0.3, 20000, 11000, 2100, false, true);
				textr = !textr;

			}
			if (tr && !windowC->getInputValue(VK_CONTROL)) {
				tr = false;
			}
			windowC->setMousePosInput(0, 0);
		}


		if (windowC->getInputValue(VK_ESCAPE)) {
			windowC->setInputValue(VK_ESCAPE, false);
			windowC->setCaptured(!windowC->isCaptured());
		}
		player_head = modSphereD(player_head, tempPos.getX(), tempPos.getY() + (windowC->getInputValue(VK_CONTROL) ? crouchHeight : baseHeight), tempPos.getZ(), 0.3, 16000, 8784, 1632, false, false);
		player_pos = modSphereD(player_pos, tempPos.getX(), tempPos.getY(), tempPos.getZ(), 0.3, 3000, 3000, 20000, false, false);
		refreshScreenValue(frameTime, !windowC->isCaptured());

		InvalidateRect(windowC->windowH, NULL, TRUE);
		UpdateWindow(windowC->windowH);
		windowC->sizing = false;
		dt2 = timeSinceEpochMillisec();
		frameTime = (int)(dt2 - dt);
		dt = dt2;
	}

	return 0;
}