#include <math.h>
#include <opencv2/core.hpp>
#include <dlib/image_processing.h>

#include "util.hpp"

using namespace std;
using namespace cv;

double Util::getAngleBetweenTwoPoints(const double y, const double x)
{
	double angle = atan2(y, x) * 180 / M_PI;
	//double angle2 = atan2(x, y) * 180 / M_PI;

	//cout << "Angle 1: " << angle << "| dif:" << 180 + angle << "| dif2: " << 180 - angle << "\n";
	//cout << "Angle 2: " << angle2 << "| dif:" << 180 + angle2 << "| dif2: " << 180 - angle2 << "\n";

	if (angle < 0)
	{
		return 360 + angle;
	}

	return angle;
}

void Util::cvRecttodlibRectangle(const Rect &r, dlib::rectangle &rec)
{
	rec = dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

double Util::getCenterOfSegment(const double lineLength, const double size){
	return lineLength + (size / 2) ;
}