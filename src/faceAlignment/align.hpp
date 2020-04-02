#ifndef FACE_ALIGNMENT_HPP
#define FACE_ALIGNMENT_HPP

#include <iostream>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

// https://docs.opencv.org/4.2.0/db/d28/tutorial_cascade_classifier.html | Detect Eyes
// https://hetpro-store.com/TUTORIALES/opencv-rect/ | Info about Rect
// https://github.com/meefik/face-alignment/blob/master/detect.js | Face alignment in js
// https://stackoverflow.com/questions/8267191/how-to-crop-a-cvmat-in-opencv | Crop image

class FaceAlignment
{
	const static string EYESMODEL;

private:
	/**
	 * Load the model and passes objects found in a Vector of Rect
	*/
	static void detectEyes(const Mat &faceROI, vector<Rect> &eyesDetected);
	/**
	 * Get coordinates of the eye center
	*/
	static void getEyeCenter(const Rect &face, const Rect &eye, Point &eyeCoordinates);
	/**
	 * Get coordinates of the face center
	*/
	static void getFaceCenter(const Rect &face, Point &faceCoordinates);
	/**
	 * Get THE POSITIVE angle between the eyes
	 * Just use two points:
	 * @link https://stackoverflow.com/questions/10143555/how-to-align-face-images-c-opencv 
	*/
	static double getAngleBetweenEyes(const Point &eyeA, const Point &eyeB);
	/**
	 * Show Mat in window
	*/
	static void showWindow(const Mat &img);
	static void drawEyes(vector<Rect> eyesDetected, const Rect &faceArea, const Mat &image);
	/**
	 * Main align method, public are variant of this
	*/
	static Mat alignFaceComplete(
			const Mat &image,
			const Rect &faceArea,
			const int height,
			const int width,
			const bool debugMode,
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
			const int width);
	/**
	 * @see FaceAlignment::alignFace()
	 * visually see the process, will not resize
	 * @param Compare if false will see the real output else you will compare rotation with original so no resize will be applied
	*/
	static Mat alignFaceDrawMode(
			const Mat &image,
			const Rect &faceArea,
			const int height,
			const int width,
			const bool compare);
};

#endif