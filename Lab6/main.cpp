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

vector<DMatch> autotune_matches(vector<DMatch> Matches, float min_dist, float ratio)
{
	vector<DMatch> match;
	for (int j = 0; j < int(Matches.size()); j++)
	{
		if (Matches[j].distance < min_dist * ratio)
			match.push_back(Matches[j]);
	}
	return match;
}

//-----------------------------------------------------functions
/*
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
*/
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
	if (argc < 5) {
		std::cout << "Error - 4 images needed" << std::endl;
		return 0;
	}

	for (int i = 1; i < 5; i++) {
		Mat img = imread(argv[i], IMREAD_COLOR);
		src.push_back(img);
		resize(img, img, Size(img.cols / 2, img.rows / 2));
		imshow("source images", img);
		waitKey(100);
	}
	destroyWindow("source images");




	/* upload video frames */
	int count = 0;
	cout << "frames uploading";
	cv::VideoCapture cap("video.mov");
	if (cap.isOpened())
	{
		for (;;) {


			cv::Mat frame;
			cap >> frame;
			if (count == 2) break;
			if (!cap.read(frame)) break;
			frames.push_back(frame);
			if(count % 100 ==0) std::cout << ".";

			/*namedWindow("video", WINDOW_FREERATIO);
			imshow("video", frame);
			*/
			waitKey(1);
			count++;

		}
	}
	else std::cout << "error 404 video not found" << std::endl;
	cap.release();
	//destroyWindow("video");

	cout << "number frames found: " << frames.size() << std::endl;

	/* get the first frame to locate features */
	Mat mainFrame = frames[0];
	Mat tmp = mainFrame;
	src.push_back(tmp);
	namedWindow("main frame", WINDOW_FREERATIO);
	imshow("main frame", mainFrame);
	cout << "feature detection: " << std::endl;

	//create the objects detector and extractor for SIFT 
	cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create();
	cv::Ptr<cv::SiftDescriptorExtractor> extractor = cv::SiftDescriptorExtractor::create();

	//variables to store the results of SIFT processing
	std::vector < std::vector<cv::KeyPoint> > keypoints;
	std::vector<cv::Mat> descriptors;

	//String indice;              ho tenuto queste due righe perche' non so se e' piu' corretto tenerle fuori dal ciclo for per inizializzarle una sola volta o tenerle(a capo)
	//Mat tmpdescriptors;		  dentro il ciclo for perche' in realta' sarebbero variabili del ciclo e quindi e' meglio tenerle dentro anche se le inizializza ad ogni iterazione.

	/* array of colors */
	std::vector <cv::Scalar> colors;
	for (int i = 0; i < 5; i++) {
		Scalar clr(rand() % 255, rand() % 255, rand() % 255);
		colors.push_back(clr);
	}

	//ciclo for per determinare keypoints e relativi descriptors dei quattro libri presi singolarmente e quando sono tutti insieme nel mainFrame
	for (int i = 0; i < 5; i++)
	{
		String indice;
		indice = to_string(i);

		Mat input = src[i];
		std::vector<cv::KeyPoint> keypoints_tmp;
		detector->detect(input, keypoints_tmp);
		keypoints.push_back(keypoints_tmp);

		cv::Mat output;
		cv::drawKeypoints(input, keypoints[i], output, colors[i]);
		//cv::imshow("image" + indice, output);
		waitKey(100);


		Mat tmpdescriptors;
		extractor->detectAndCompute(input, Mat(), keypoints[i], tmpdescriptors);
		descriptors.push_back(tmpdescriptors);
		//imshow("ff" + indice, descriptors[i]);

		//stavo/sto  cercando di usare BFMatcher come l'esempio a questo sito:https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html

		/*Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
		std::vector< std::vector<DMatch> > knn_matches;
		matcher->knnMatch(descriptors, knn_matches, 2);
		*/

		
	}

	/* COPIATO DA ANIELLO */
	vector<DMatch> tmp_matches;
	vector<vector<DMatch>> match;


	// 4 should be the norm-type ENUM that correspond to NORM_HAMMING --- we use also cross-match
	/*	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2, false);

	for (int i = 0; i < descriptors.size() - 1; i++)
	{
		//cout << "Size of " << i << "-th descriptor: " << Descriptors[i].size() << endl;

		matcher->match(descriptors[i], descriptors[4], tmp_matches, Mat());

		match.push_back(tmp_matches);
	}
	*/


	//Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<DMatch> > knn_matches;
	//matcher->knnMatch(descriptors[0], descriptors[4], knn_matches, 2);

	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2, true);
	matcher->match(descriptors[0], descriptors[4], tmp_matches, Mat());
	knn_matches.push_back(tmp_matches);


	std::vector<KeyPoint> keypoints_object, keypoints_scene;

	keypoints_object = keypoints[0];
	keypoints_scene = keypoints[4];
	Mat img_object = src[0].clone();
	Mat img_scene = src[4].clone();;


	std::vector<DMatch> good_matches;
	/* COPIATO DA SITO DI GIACOMO */

	/*
	//-- Filter matches using the Lowe's ratio test
	const float ratio_thresh = 0.75f;
	
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}*/
	float ratio = 1.5;

	float instance_ratio = ratio;

	// push in bestMatches the top 50 matched features
	vector<DMatch> bestMatches;
	int nbMatch = int(knn_matches[0].size());
	Mat tab(nbMatch, 1, CV_32F);

	float dist;
	float min_dist = -1.;

	// Find the minumun distance between matchpoints
	for (int j = 0; j < nbMatch; j++)
	{
		dist = knn_matches[0][j].distance;

		// update the minumun distance
		if (min_dist < 0 || dist < min_dist)
			min_dist = dist;
	}


	// Adapt the ratio in order to get at least 120 matches per couple of adjacent images
	do
	{
		good_matches = autotune_matches(knn_matches[0], min_dist, instance_ratio);
		instance_ratio = 2 * instance_ratio;
	} while (good_matches.size() < 120);



	//cout << "Size of " << i << "-th" << "TOP matches: " << BestMatches[i].size() << endl;




	//-- Draw matches
	Mat img_matches;
	drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, colors[0],
		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}
	cout << "good_matches size:     " << good_matches.size() << std::endl;
	cout << "keypoints_object size:     " << keypoints_object.size() << std::endl;
	cout << "obj size:     "<< obj.size()<<std::endl;
	Mat H = findHomography(obj, scene, RANSAC);
	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point2f(0, 0);
	obj_corners[1] = Point2f((float)img_object.cols, 0);
	obj_corners[2] = Point2f((float)img_object.cols, (float)img_object.rows);
	obj_corners[3] = Point2f(0, (float)img_object.rows);
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H);
	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
		scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
		scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
		scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
		scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
	//-- Show detected matches
	namedWindow("Good Matches & Object detection", WINDOW_AUTOSIZE);
	resize(img_matches, img_matches, Size(img_matches.cols / 2, img_matches.rows / 2));
	imshow("Good Matches & Object detection", img_matches);


	cout << "END" << std::endl;



	waitKey(0);
	return 0;
}

//------------------------------------------------------------------------------------
//                                FUNCTIONS
//------------------------------------------------------------------------------------

