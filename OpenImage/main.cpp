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

	/* create a vector of images to find and track and a vector for the video frames*/

	std::vector<cv::Mat> src;
	std::vector<cv::Mat> frames;

	/* upload images to track */  
	// ADJUST THIS CHECK....WE CAN HAVE 1..2...3..4..5.. IMAGES...
	// FIND A WAY TO KNOW HOW MANY IMAGES ARE PASSED THRUOGHT THE COMMAND LINE
	// AND USE THIS NUMBER AS SUPERIOR LIMIT TO THE FOR CYCLE BELOW
	if (argc < 5 ) {
		std::cout << "Error - 4 images needed" << std::endl;
		return 0;
	}

	for (int i = 1; i < 5; i++) {
		Mat img = imread(argv[i], IMREAD_COLOR);
		src.push_back(img);
		resize(img, img, Size(img.cols / 2, img.rows / 2));
		imshow("source images", img);
		waitKey(500);
	}
	destroyWindow("source images");
	

	/* upload video fraames */
	cout << "framse uploading" << std::endl;
	cv::VideoCapture cap("video.mov");
		if (cap.isOpened()) {
			for (;;) {
				cv::Mat frame;
				cap >> frame;
				frames.push_back(frame);
				if (!cap.read(frame)) break;
			}
		}
	cout <<"number frames found: "<< frames.size() << std::endl;
	
	waitKey(0);
	return 0;
}

//------------------------------------------------------------------------------------
//                                FUNCTIONS
//------------------------------------------------------------------------------------


