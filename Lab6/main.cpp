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
		waitKey(500);
	}
	destroyWindow("source images");


	

	/* upload video frames */
	cout << "frames uploading..." << std::endl;
	cv::VideoCapture cap("C:/workspace/Lab6/video.mov");
	if (cap.isOpened())
	{
		for (;;) {


			cv::Mat frame;
			cap >> frame;

			if (!cap.read(frame)) break;
			frames.push_back(frame);

			/*namedWindow("video", WINDOW_FREERATIO);
			imshow("video", frame);
			*/
			waitKey(1);
			

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
	std::vector<cv::KeyPoint> keypoints;
	std::vector<cv::Mat> descriptors;
	
	//String indice;              ho tenuto queste due righe perche' non so se e' piu' corretto tenerle fuori dal ciclo for per inizializzarle una sola volta o tenerle(a capo)
	//Mat tmpdescriptors;		  dentro il ciclo for perche' in realta' sarebbero variabili del ciclo e quindi e' meglio tenerle dentro anche se le inizializza ad ogni iterazione.
	
	//ciclo for per determinare keypoints e relativi descriptors dei quattro libri presi singolarmente e quando sono tutti insieme nel mainFrame
	for (int i = 0; i <5; i++)
	{
		String indice;
		indice = to_string(i);
		
		Mat input= src[i];
		detector->detect(input, keypoints);

		cv::Mat output;
		cv::drawKeypoints(input, keypoints, output);
		cv::imshow("image" + indice, output);
		waitKey(500);
		
		
		Mat tmpdescriptors;
		extractor->detectAndCompute(input, Mat(), keypoints, tmpdescriptors);
		descriptors.push_back(tmpdescriptors);
		imshow("ff" + indice, descriptors[i]);
		
		//stavo/sto  cercando di usare BFMatcher come l'esempio a questo sito:https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
		/*Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
		std::vector< std::vector<DMatch> > knn_matches;
		matcher->knnMatch(descriptors, knn_matches, 2);
		*/

	}
	

	
	cout << "END" << std::endl;


	
	waitKey(0);
	return 0;
}

//------------------------------------------------------------------------------------
//                                FUNCTIONS
//------------------------------------------------------------------------------------


