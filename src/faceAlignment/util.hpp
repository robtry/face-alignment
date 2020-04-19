#ifndef UTIL_HPP
#define UTIL_HPP

#include <opencv2/core.hpp>
#include <dlib/image_processing.h>

class Util
{
public:
	/**
	 * Returns the positive angle between two points
	 * if rotation is counter clockwise 360 + angle which is negative
	 * otherwise return angle which is positive
	*/
	static double getAngleBetweenTwoPoints(const double y, const double x);
	/**
	 * Convert opencv Rect in to dlib rectangle
	*/
	static void cvRecttodlibRectangle(const cv::Rect &r, dlib::rectangle &rec);
	static double getCenterOfSegment(const double lineLength, const double size);
};

#endif