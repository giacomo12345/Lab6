/* COMPUTER VISION - LAB 6 - AMBROSIN GIOELE - MULTINEDDU GIACOMO */
/* classes  myObject and myMatcher*/
/* object myObject includes the image, the keypoints, the descriptors and the color associate to the object */
/* class myMatcher takes the vector of objects to be recogized inside the object scene */

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

////////////////////////////////////////////////////////////////////////
//							 myObject
////////////////////////////////////////////////////////////////////////
class myObject {

public:

	/* constructors */
	myObject();
	myObject(Mat src);

	/* show the object source image */
	void showImage();

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


////////////////////////////////////////////////////////////////////////
//							myMatcher
////////////////////////////////////////////////////////////////////////
class myMatcher {

public:

	/* constructor */
	myMatcher(std::vector<myObject> objects, myObject first_frame);

	/* compute the match between objects and scene image */
	void computeMatches();

	/* select the matches according to the distance */ 
	void filterMatches(float ratio, int features_min);

	/* return a vector of images where each keypoints of object are connected to the matched scene keypoints */
	std::vector<cv::Mat> getImageMatched();

	/* localize object and  compute homography  */
	void computeHomography();

	/* find the object images corners to draw the rettangle */
	void findCorners();

	/* compute the projection from the object images to the scene image*/
	void computeProjection();

	/* return the scene points */
	std::vector < std::vector<Point2f> >getScenePoints();
	/* return the corners points */
	std::vector<std::vector<Point2f>> getSceneCorners();

	
private:

	std::vector<myObject> obj;
	myObject scene;
	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2, true);
	std::vector< std::vector<DMatch> > matches;
	std::vector<std::vector<DMatch>> good_matches;
	std::vector<std::vector<Point2f>> obj_points;
	std::vector<std::vector<Point2f>> scene_points;
	std::vector<std::vector<Point2f>> obj_corners;
	std::vector<std::vector<Point2f>> scene_corners;
	std::vector<Mat> H;
	
};