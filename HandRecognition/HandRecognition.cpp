#include "pch.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

RNG rng(12345);

/**  @function main */
int main(int argc, char** argv)
{
	Mat src;
	/// Load image
	src = imread("C:\\Users\\Nuno\\Pictures\\hand.png");

	for (int i = 1; i < 25; i = i + 2)
	{
		GaussianBlur(src, src, Size(i, i), 0, 0);
	}

	Mat ycrcb; cvtColor(src, ycrcb, COLOR_RGB2YCrCb);

	vector<Mat> spl; split(src, spl);

	//spl[0] = 0;
	spl[1] = 0;
	spl[2] = 0;

	Mat final; merge(spl, final);

	cvtColor(final, final, COLOR_YCrCb2RGB);
	cvtColor(final, final, COLOR_RGB2GRAY);
	
	threshold(final, final, 127, 255, THRESH_BINARY | THRESH_OTSU);

	Mat canny;
	Canny(final, canny, 127, 255, 3);

	vector<vector<Point>> contours; // Vector for storing contour
	vector<Vec4i> hierarchy;

	findContours(canny, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image

	double largestArea = 0;
	int index = 0;
	Rect rect;

	for (int i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i]);
		if (area > largestArea) {
			largestArea = area;
			index = i;
			rect = boundingRect(contours[i]);
		}
	}

	drawContours(canny, contours, index, Scalar(255, 0, 0), CV_FILLED, 8, hierarchy);
	rectangle(canny, rect, Scalar(255, 0, 0), 1, 8, 0);

	imshow("Contour", canny);

	/// Wait until user exits the program
	waitKey(0);

	return 0;
}