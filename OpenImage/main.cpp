/* COMPUTER VISION - LAB 6 - AMBROSIN GIOELE - MULTINEDDU GIACOMO*/

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/base.hpp>
#include <iostream>


using namespace cv;
using namespace std;

//-----------------------------------------------------functions

//-----------------------------------------------------variables




//------------------------------------------------------------------------------------
//                               MAIN FUNCTION
//------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {

	if (argc < 2) {
		std::cout << "Error - an image is needed" << std::endl;
		return 0;
	}
	Mat img = imread(argv[1], IMREAD_COLOR);
	resize(img, img, Size(img.cols / 2, img.rows / 2));
	imshow("test window", img);

	cv::VideoCaptu
