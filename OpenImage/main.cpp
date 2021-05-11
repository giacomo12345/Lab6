#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
	cv::Mat img = cv::imread("immagine.png");
	cv::namedWindow("Example1");
	cv::imshow("Example1", img);
	cv::waitKey(0);

	return 0;
}




