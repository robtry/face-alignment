#include <math.h>
#include <opencv2/core.hpp>
#include <dlib/image_processing.h>

#include "util.hpp"

using namespace std;
using namespace cv;

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

void Util::cvRecttodlibRectangle(const Rect &r, dlib::rectangle &rec)
{
	rec = dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}