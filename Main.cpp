#include "kernel_wrapper.cuh"
#include "Header.h"
#include <tchar.h>
#include <Windows.h>
#include <Windowsx.h>
#include <WinUser.h>
#include "Client.cpp"
#include "WindowsHolder.h"

int height, width;
WindowsHolder* windowW;
HCURSOR cursorBase, cursorHide;

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

void resizeClient(int Nwidth, int Nheight) {
	if (Nwidth != width || Nheight != height) {
		height = Nheight;
		width = Nwidth;
		if (windowW) {
			windowW->setDrawSize(width, height);
		}
		
	}
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR pCmdLine, int nCmdShow)
{
	// Register the window class.
	const char CLASS_NAME[] = _T("RayTracingMain");
	cursorBase = LoadCursor(NULL, IDC_HAND);
	cursorHide = LoadCursor(NULL, NULL);
	WNDCLASS wc = { };

	wc.style = CS_HREDRAW | CS_VREDRAW;
	wc.lpfnWndProc = (WNDPROC)WindowProc;
	wc.hInstance = hInstance;
	//wc.hCursor = cursorBase;
	wc.lpszClassName = CLASS_NAME;

	RegisterClass(&wc);

	// Create the window.

	HWND hwnd = CreateWindowEx(
		0,                              // Optional window styles.
		CLASS_NAME,                     // Window class
		"Ray Tracing",    // Window text
		WS_OVERLAPPEDWINDOW,            // Window style

		// Size and position
		100, 100, 720, 405,

		NULL,       // Parent window    
		NULL,       // Menu
		hInstance,  // Instance handle
		NULL        // Additional application data
	);

	if (hwnd == NULL)
	{
		return 0;
	}

	ShowWindow(hwnd, nCmdShow);

	
	// Run the message loop.
	RECT rect;
	GetClientRect(hwnd, &rect);
	height = rect.bottom - rect.top;
	width = rect.right - rect.left;
	windowW = new WindowsHolder(hwnd);
	windowW->setDrawSize(width, height);

	DWORD clientThreadID;
	HANDLE clientHandle = CreateThread(0, 0, clientMain, windowW, 0, &clientThreadID);

	MSG msg = { };
	while (GetMessage(&msg, NULL, 0, 0))
	{
		if (windowW->exitWindow) {
			windowW->exitWindow = false;
			CloseHandle(clientHandle);
			DestroyWindow(hwnd);
		}
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	return 0;
}

LRESULT APIENTRY WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg)
	{
	case WM_DESTROY:
	{
		PostQuitMessage(0);
		break;
	}
	case WM_KEYDOWN:
	{
		char keyCode = (char)wParam;
		windowW->setInputValue(keyCode, true);
		break;
	}
	case WM_KEYUP:
	{
		char keyCode = (char)wParam;
		windowW->setInputValue(keyCode, false);
		break;
	}
	case WM_SYSKEYDOWN:
	{
		char keyCode = (char)wParam;
		windowW->setInputValue(keyCode, true);
		return DefWindowProcA(hwnd, uMsg, wParam, lParam);
	}
	case WM_SYSKEYUP:
	{
		char keyCode = (char)wParam;
		windowW->setInputValue(keyCode, false);
		break;
	}
	case WM_ERASEBKGND:
	{
		return false;
	}
	case WM_PAINT:
	{

		PAINTSTRUCT ps;
		HDC hdc = BeginPaint(hwnd, &ps);

		int* screen = windowW->getOutputScreen();
		if (!windowW->sizing && screen) {
			int maxX = screen[0];
			int maxY = screen[1];

			COLORREF *colors = (COLORREF*)calloc(maxX*maxY, sizeof(COLORREF));
			memcpy(colors, screen + 2, maxX*maxY * sizeof(int));
			screen = windowW->getOutputScreen();
			if (screen &&
				screen[0] &&
				screen[1] == maxY) {
				HBITMAP map = CreateBitmap(maxX, maxY, 1, 8 * 4, (void*)colors);
				HDC src = CreateCompatibleDC(hdc);
				SelectObject(src, map);
				BitBlt(hdc, 0, 0, maxX, maxY, src, 0, 0, SRCCOPY);
				free(colors);
				DeleteObject(map);
				DeleteDC(src);
			}
			else {
				FillRect(hdc, &ps.rcPaint, (HBRUSH)(COLOR_DESKTOP));
			}
		}
		else {
			FillRect(hdc, &ps.rcPaint, (HBRUSH)(COLOR_DESKTOP));
		}

		EndPaint(hwnd, &ps);
		break;
	}
	case WM_SIZE:
	{
		RECT rect;
		GetClientRect(hwnd, &rect);
		int tempW = rect.right - rect.left;
		int tempH = rect.bottom - rect.top;
		resizeClient(tempW, tempH);
		break;
	}
	case WM_SIZING:
	{
		windowW->sizing = true;
		RECT rect;
		GetClientRect(hwnd, &rect);
		int tempW = rect.right - rect.left;
		int tempH = rect.bottom - rect.top;
		resizeClient(tempW, tempH);
		break;
	}
	case WM_MOUSEMOVE:
	{
		if (windowW->isCaptured()) {

			RECT rect = {};
			POINT mPos = {};
			GetCursorPos(&mPos);
			GetWindowRect(hwnd, &rect);
			int tempX = rect.left + (width / 2);
			int tempY = rect.top + (height / 2);
			if (mPos.x != tempX || mPos.y != tempY) {
				int rPosX = mPos.x - tempX;
				int rPosY = mPos.y - tempY;
				windowW->setMousePosInput(rPosX + windowW->getMousePosX(), rPosY + windowW->getMousePosY());
				SetCursorPos(tempX, tempY);
			}
		}
		else {
			windowW->setMousePosInput(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
		}
		break;
	}
	case WM_SETCURSOR:
	{
		if (windowW->isCaptured()) {
			SetCursor(cursorHide);
		}
		else {
			SetCursor(cursorBase);
		}
		break;
	}
	return 0;

	}
	return DefWindowProc(hwnd, uMsg, wParam, lParam);
}