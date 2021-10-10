#include "GuiHelper.h"

int checkCom(unsigned char* data, int index) {
	int l = index;
	while (data[l] == 0x23) {
		while (data[l] != 0x0a) {
			l++;
		}
		l++;
	}
	return l;
}

int charToInt(unsigned char* data, int* index) {
	int res = 0;
	while (data[*index] <= 0x39 && data[*index] >= 0x30) {
		res *= 10;
		res += ((int)data[*index] - 48);
		(*index) = (*index) + 1;
	}
	return res;
}

char* intToChar(int nb) {
	char* res = new char[32];
	char* tmp = new char[32];
	int nbt = nb;
	int ind = 0;
	bool neg = false;
	if (nbt < 0) {
		nbt = -nbt;
		neg = true;
	}
	if (nbt == 0) {
		ind = 1;
		tmp[0] = '0';
	}
	while (nbt > 0) {
		tmp[ind] = (char)((nbt % 10)+48);
		nbt = nbt / 10;
		ind++;
	}
	ind--;
	int cnt = 0;
	if (neg) {
		cnt = 1;
		res[0] = '-';
	}
	while (ind >= 0 && cnt < 31) {
		res[cnt] = tmp[ind];
		cnt++;
		ind--;
	}
	res[cnt] = '\0';
	delete[] tmp;
	return res;
}

void drawString(int* screen, const char* txt, int posX, int posY, int color) {
	int* sc = screen + 2;
	int ind = 0;
	if (!chars || !screen || !txt) {
		return;
	}
	while (txt[ind] != 0x0) {
		for (int i = 0; i < 64; i++) {
			int ps = screen[0] * (i / 8 + posY) + posX + i % 8 + ind*8;
			if (chars[(unsigned char)txt[ind]][i] == true) {
				sc[ps] = color;
			}
		}
		ind++;
	}
}

void drawTex(int* screen, unsigned char* tex, int posX, int posY) {
	int* sc = screen + 2;
	if (!tex || !screen) {
		return;
	}
	int ssz = screen[0] * screen[1];
	int sX = (int)(tex[0]) * 256 + (int)(tex[1]);
	int sY = (int)(tex[2]) * 256 + (int)(tex[3]);
	for (int i = 0; i < sX*sY; i++) {
		int p1 = posX + i % sX;
		if (p1 < screen[0]) {
			int ps = screen[0] * (i / sX + posY) + p1;
			if (ps >= ssz) {
				break;
			}
			int id = 3 * i + 4;
			sc[ps] = (int)(tex[id + 2]) + ((int)(tex[id + 1]) << 8) + ((int)(tex[id]) << 16);
		}
	}
}

int loadTexture(unsigned char** tex, const char* path) {
	FILE* cr;
	long lSize;
	fopen_s(&cr, path, "rb");
	if (cr == NULL) {
		return 1;
	}
	fseek(cr, 0, SEEK_END);
	lSize = ftell(cr);
	if (lSize < 10) {
		return 1;
	}
	rewind(cr);
	unsigned char* t = new unsigned char[lSize];
	fread(t, 1, lSize, cr);

	if (t[1] != '6') {
		return 2;
	}
	int l = checkCom(t, 3);
	long long int x = charToInt(t, &l);
	l++;
	long long int y = charToInt(t, &l);
	l++;
	long long int max = charToInt(t, &l);
	l++;
	int st = 0;
	unsigned char* texture = new unsigned char[x*y*3+4];
	texture[0] = (unsigned char)(x/256);
	texture[1] = (unsigned char)(x%256);
	texture[2] = (unsigned char)(y/256);
	texture[3] = (unsigned char)(y%256);
	while (st < x*y && l < lSize-2) {
		int ind = 3*st+4;
		texture[ind] = t[l];
		texture[ind+1] = t[l+1];
		texture[ind+2] = t[l+2];
		l = l + 3;
		st++;
	}
	*tex = texture;
	return 0;
}

int initChars()
{
	FILE* cr;
	long lSize;
	fopen_s(&cr, "./pictures/table.ppm", "rb");
	if (cr == NULL) {
		return 1;
	}
	fseek(cr, 0, SEEK_END);
	lSize = ftell(cr);
	if (lSize < 10) {
		return 10;
	}
	rewind(cr);
	unsigned char* t = new unsigned char[lSize];
	fread(t,1,lSize,cr);
	
	if (t[1] != '6') {
		return 2;
	}
	int l = checkCom(t,3);
	int x = charToInt(t,&l);
	l++;
	int y = charToInt(t, &l);
	l++;
	int max = charToInt(t, &l);
	l++;
	chars = new bool*[256];
	charSX = (x/16);
	charSY = (y/16);
	int st = 0;
	if (charSY < 4 || charSX < 4) {
		return 3;
	}
	for (int i = 0; i < 256; i++) {
		chars[i] = new bool[(__int64)charSX*(__int64)charSY];
	}
	while (st < x*y) {
		int cY = (st / x)%charSY;
		int cX = st % charSX;
		int car = (st / (x*charSY)) * 16 + (st % x) / charSX;
		chars[car][cY*charSX+cX] = ((int)(t[l]) == max?false:true);
		l = l + 3;
		st++;
	}
	return 0;
}

void endChars() {
	for (int i = 0; i < 256; i++) {
		delete[] chars[i];
	}
	delete[] chars;
	chars = 0;
}

void renderGUI(int* screen, int frameTime, bool menu) {
	drawString(screen, "Frame Time:", 10, 10, 0x00000000);
	char* tm = intToChar(frameTime);
	drawString(screen, tm, 106, 10, 0x0000ff00);
	delete[] tm;
	if (menu) {
		drawString(screen, "PAUSE", 10,70,0x007f0000);
	}
}