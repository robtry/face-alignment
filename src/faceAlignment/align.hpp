#ifndef FACE_ALIGNMENT_HPP
#define FACE_ALIGNMENT_HPP

#include <iostream>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

class FaceAlignment
{
	const static string EYESMODEL;

private:
	/**
	 * returns true if can detect exactly 2 eyes
	*/
	static void detectEyes(const Mat &face, vector<Rect> &eyesDetected);
	/**
	 * Get coordinates of center
	*/
	static void getEyeCenter(const Rect &face, const Rect &eye, Point &eyeCoordinates);
	/**
	 * Get coordinates of the face center
	*/
	static void getFaceCenter(const Rect &face, Point &faceCoordinates);
	/**
	 * Returns angle between two points
	*/
	static double getAngleBetweenEyes(const Point &eyeA, const Point &eyeB);
	/**
	 * Get coordinates of center
	*/
	static void rotate(Mat &face);
	/**
	 * Main align method, public are variant of this
	*/
	static Mat alignFaceComplete(
			const Mat &image,
			const Rect &faceArea,
			const int height,
			const int width,
			const bool debugMode,
			const bool deepDebugMode,
			const bool drawMode);

public:
	/**
	 * @param Image => current image analyzing
	 * @param FaceArea => current ROI, where is possible to detect eyes
	 * @param Height => for output align image
	 * @param Width => for output align image
	 * 
	 * Align the face using eyes as reference
	 * @return cv::Mat in grayscale, aligned and cropped
	*/
	static Mat alignFace(
			const Mat &image,
			const Rect &faceArea,
			const int height,
			const int width);
	/**
	 * @see FaceAlignment::alignFace()
	 * @param Debug => output details
	 * Print details
	*/
	static Mat alignFaceDebugMode(
			const Mat &image,
			const Rect &faceArea,
			const int height,
			const int width,
			const bool deepDebug);
	/**
	 * @see FaceAlignment::alignFace()
	 * visually see process
	*/
	static Mat alignFaceDrawMode(
			const Mat &image,
			const Rect &faceArea,
			const int height,
			const int width);
};

#endif