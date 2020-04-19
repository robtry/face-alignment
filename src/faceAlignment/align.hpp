#ifndef FACE_ALIGNMENT_HPP
#define FACE_ALIGNMENT_HPP

#include <iostream>
#include <opencv2/core.hpp>
#include <dlib/image_processing.h>											 //shape_predictor
#include <dlib/image_processing/frontal_face_detector.h> // frontal face

// http://dlib.net/face_landmark_detection_ex.cpp.html | Landmark detection
// https://hetpro-store.com/TUTORIALES/opencv-rect/ | Info about Rect
// https://stackoverflow.com/questions/8267191/how-to-crop-a-cvmat-in-opencv | Crop image

class FaceAlignment
{

private:
	/** path to the model*/
	std::string faceLandmarkModel;

	/** store landmarks model provided by dlib */
	dlib::shape_predictor landmarkDetector;

	/** to store the default face detector method */
	dlib::frontal_face_detector faceDetector;

	/** 
	 * Load the model using deserialize provided by dlib,
	 * and start faceDetector provided by dlib,
	 * this should be called inside constructor
	*/
	void loadModelAndStartDetector();

	/**
	 * 
	*/
	void getFaceLandmarks(
			const cv::Mat &image,
			std::vector<cv::Point2f> &points,
			dlib::shape_predictor landmarkDetector);
	// /**
	// * Load the model and passes objects found in a Vector of Rect
	// */
	// void detectEyes(const Mat &faceROI, vector<Rect> &eyesDetected);
	// /**
	//  * Get coordinates of the eye center
	// */
	// void getEyeCenter(const Rect &face, const Rect &eye, Point &eyeCoordinates);
	// /**
	//  * Get coordinates of the face center
	// */
	// void getFaceCenter(const Rect &face, Point &faceCoordinates);
	// /**
	//  * Get THE POSITIVE angle between the eyes
	//  * Just use two points:
	//  * @link https://stackoverflow.com/questions/10143555/how-to-align-face-images-c-opencv
	// */
	// double getAngleBetweenEyes(const Point &eyeA, const Point &eyeB);
	// /**
	//  * Show Mat in window
	// */
	// void showWindow(const Mat &img);
	// void drawEyes(vector<Rect> eyesDetected, const Rect &faceArea, const Mat &image);
	/**
	 * Main align method, public are variant of this
	*/
	cv::Mat alignFaceComplete(
			const cv::Mat &image,
			const cv::Rect &faceArea,
			const int height,
			const int width,
			const bool debugMode,
			const bool drawMode);

public:
	/** Default constructor */
	FaceAlignment();
	/**
	 * @param path => route to shape_predictor_68_face_landmarks.dat
	 * @link http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
	*/
	FaceAlignment(const std::string path);
	/**
	 * @param Image => current image analyzing
	 * @param FaceArea => current ROI, where is possible to detect eyes
	 * @param Height => for output align image
	 * @param Width => for output align image
	 * 
	 * Align the face using eyes as reference
	 * @return cv::Mat in grayscale, aligned and cropped
	*/
	cv::Mat alignFace(
			const cv::Mat &image,
			const cv::Rect &faceArea,
			const int height,
			const int width);
	// /**
	//  * @see FaceAlignment::alignFace()
	//  * @param Debug => output details
	//  * Print details
	// */
	// Mat alignFaceDebugMode(
	// 		const Mat &image,
	// 		const Rect &faceArea,
	// 		const int height,
	// 		const int width);
	// /**
	//  * @see FaceAlignment::alignFace()
	//  * visually see the process, will not resize
	//  * @param Compare if false will see the real output else you will compare rotation with original so no resize will be applied
	// */
	// Mat alignFaceDrawMode(
	// 		const Mat &image,
	// 		const Rect &faceArea,
	// 		const int height,
	// 		const int width,
	// 		const bool compare);
};

#endif