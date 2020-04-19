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

	/** 
	 * Load the model using deserialize provided by dlib,
	 * this should be called inside constructor
	*/
	void loadModel();

	/**
	 * Get vector of cv::Points with all landmarks
	*/
	void getFaceLandmarks(
			const cv::Mat &image,
			const cv::Rect &faceArea,
			std::vector<cv::Point> &points);

	/**
	* Get coordinate of each eye
	*/
	void getEyesCoordinates(
			const std::vector<cv::Point> &faceLandMarkPoints,
			cv::Point &leftEye,
			cv::Point &rightEye);

	/**
	 * Get coordinates of the face center
	 * helps with the rotation
	*/
	void getFaceCenter(const cv::Rect &face, cv::Point &faceCoordinates);

	// /**
	//  * Get THE POSITIVE angle between the eyes
	//  * Just use two points:
	//  * @link https://stackoverflow.com/questions/10143555/how-to-align-face-images-c-opencv
	// */
	double getAngleBetweenEyes(const cv::Point &eyeA, const cv::Point &eyeB);

	/**
	 * Show Mat in window
	 * used for debug
	*/
	void showWindow(const cv::Mat &img);

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
	/**
	 * @see FaceAlignment::alignFace()
	 * @param draw => if true will show the final image
	*/
	cv::Mat alignFaceDebugMode(
			const cv::Mat &image,
			const cv::Rect &faceArea,
			const int height,
			const int width,
			const bool draw);
};

#endif