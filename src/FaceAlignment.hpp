#ifndef FACE_ALIGNMENT_HPP
#define FACE_ALIGNMENT_HPP

#include <iostream>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

// Paths to the model
const string FaceAlignment::FACEMODEL = "/opt/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
const string FaceAlignment::EYESMODEL = "/opt/opencv/data/haarcascades/haarcascade_eye.xml";

class FaceAlignment
{
private:
	static const string FACEMODEL;
	static const string EYESMODEL;
	static void detectEyes(Mat face);
	static Point getEyeCenter(const Mat &face, const Mat &eye);
	static Point getFaceCenter(const Mat &face);
	static double getAngleBetweenEyes(const Mat &eyeA, const Mat &eyeB);
	static void rotate(Mat &face);
	static void crop(const int height, const int width);
public:
	static Mat alignFace(const vector<Rect> faces, const int height, const int width, const bool debug);
};

#endif