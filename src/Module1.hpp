#ifndef MODULE_ONE_HPP
#define MODULE_ONE_HPP

#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

class Module1 {
	private:
		static string readImage(const string file);
		
	public:
		static void detectFaces(Mat image);
};

#endif