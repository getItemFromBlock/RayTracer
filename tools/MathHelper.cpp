#include "MathHelper.h"


double prng() {
	s1 = (171 * s1) % 30269;
	s2 = (172 * s2) % 30307;
	s3 = (170 * s3) % 30323;
	return std::fmod((s1 / 30269.0 + s2 / 30307.0 + s3 / 30323.0), 1.0);
}
/*
short random_s()
{
	short temp1, temp2;
	if (seed == 22026) {
		seed = 0;
	}
	temp1 = (seed & 0x00ff) << 8;
	temp1 = temp1 ^ seed;

	seed = ((temp1 & 0x00ff) << 8) + ((temp1 & 0xff00) >> 8);

	temp1 = ((temp1 & 0x00ff) << 1) ^ seed;
	temp2 = (temp1 >> 1) ^ 0xff80;

	if ((temp1 & 1) == 0) {
		if (temp2 == 43605) {
			seed = 0;
		}
		else {
			seed = temp2 ^ 0x1ff4;
		}
	}
	else {
		seed = temp2 ^ 0x8180;
	}

	return seed;
}
*/