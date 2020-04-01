#include <math.h>
#include "util.hpp"

using namespace std;

double Util::getAngleBetweenTwoPoints(const double y, const double x){
	double angle = atan2(y , x) * 180 / M_PI;
		if(angle < 0){
			return 180.0 + angle;
		}
		return angle;
}

double Util::getCenterOfSegment(const double lineLength, const double size){
	return lineLength + (size / 2) ;
}