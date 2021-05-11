/* COMPUTER VISION - LAB 6 - AMBROSIN GIOELE - MULTINEDDU GIACOMO*/

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/base.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>


using namespace cv;
using namespace std;

//-----------------------------------------------------functions
void compute_match(vector<Mat> Descriptors, vector<vector<DMatch>>& Matches)
{

	vector<DMatch> tmp_matches;

	// 4 should be the norm-type ENUM that correspond to NORM_HAMMING --- we use also cross-match
	Ptr<BFMatcher> matcher = BFMatcher::create(4, true);

	for (int i = 0; i < Descriptors.size() - 1; i++)
	{
		//cout << "Size of " << i << "-th descriptor: " << Descriptors[i].size() << endl;

		matcher->match(Descriptors[i], Descriptors[i + 1], tmp_matches, Mat());

		Matches.push_back(tmp_matches);
		cout << "Match between " << i + 1 << "-" << i + 2 << " computed." << endl;
	}

	// Add last-first match
	matcher->match(Descriptors.back(), Descriptors[0], tmp_matches, Mat());
	Matches.push_back(tmp_matches);
	cout << "Match between last-first computed." << endl;
	return;
}
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
	

	/* upload video frames */
	cout << "frames uploading..." << std::endl;
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

	/* get the first frame to locate features */
	Mat mainFrame = frames[0];
	imshow("main frame", mainFrame);

	Mat input = src[0];
	cout << "feature detection: " <<  std::endl;

	cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create();

	std::vector<cv::KeyPoint> keypoints;

	detector->detect(input, keypoints);

	cv::Mat output;
	cv::drawKeypoints(input, keypoints, output);
	cv::imshow("sift_result.jpg", output);

	Mat descriptors;

	cv::Ptr<cv::SiftDescriptorExtractor> extractor = cv::SiftDescriptorExtractor::create();

	extractor->detectAndCompute (input, Mat(), keypoints, descriptors);
	imshow("ff", descriptors);

	cout << "END" <<  std::endl;
	
	

	waitKey(0);
	return 0;
}

//------------------------------------------------------------------------------------
//                                FUNCTIONS
//------------------------------------------------------------------------------------


