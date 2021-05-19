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

class myObject {

public:

	/* constructor */
	myObject();
	myObject(Mat src);

	/* show the object source image */
	void showImage();

	/* compute keypoints and descriptors */
	//void computeKeypointsAndDescriptors();

	/* compute keypoints and descriptors */
	void showKeypoints(String windowName);

	/* return keypoints */
	std::vector<cv::KeyPoint> getKeypoints();

	/* return descriptors */
	Mat getDescriptors();

	Mat image;
	Scalar color;

private:

	/* objects detector and extractor for SIFT */
	cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create();
	cv::Ptr<cv::SiftDescriptorExtractor> extractor = cv::SiftDescriptorExtractor::create();

	Mat descriptors;
	std::vector<cv::KeyPoint> keypoints;

};

class myMatcher {

public:

	/* constructor */
	myMatcher(std::vector<myObject> objects, myObject first_frame);

	/* compute the match between objects and scene image */
	void computeScene();

private:

	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2, true);
	std::vector< std::vector<DMatch> > matches;
	std::vector<myObject> obj;
	myObject scene;
	

};