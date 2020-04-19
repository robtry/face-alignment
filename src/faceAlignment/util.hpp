#ifndef UTIL_HPP
#define UTIL_HPP

#include <opencv2/core.hpp>
#include <dlib/image_processing.h>

class Util {
	public:
		static double getAngleBetweenTwoPoints(const double y, const double x);
		static double getCenterOfSegment(const double lineLength, const double size);
		static void cvRecttodlibRectangle(const cv::Rect &r, dlib::rectangle &rec);
};

#endif