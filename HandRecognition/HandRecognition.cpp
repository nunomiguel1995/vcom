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

	Mat ycrcb; cvtColor(src, ycrcb, COLOR_RGB2YCrCb);

	vector<Mat> spl; split(src, spl);

	spl[1] = 0;
	spl[2] = 0;

	Mat final; merge(spl, final);

	cvtColor(final, final, COLOR_YCrCb2RGB);
	cvtColor(final, final, COLOR_RGB2GRAY);
	
	GaussianBlur(final, final, Size(5, 5), 0, 0);
	threshold(final, final, 100, 255, THRESH_BINARY | THRESH_OTSU);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(final, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image

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

	vector<vector<Point>>hull(contours.size());
	vector<vector<int> > hullsI(contours.size()); // Indices to contour points
	vector<vector<Vec4i>>defects(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		convexHull(contours[i], hull[i], false);
		convexHull(contours[i], hullsI[i], false);
		if (hullsI[i].size() > 3) {
			convexityDefects(contours[i], hullsI[i], defects[i]);
		}
	}

	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		drawContours(final, contours, (int)i, color);
		drawContours(final, hull, (int)i, color);
	}
	
	imshow("Contour", final);

	/// Wait until user exits the program
	waitKey(0);

	return 0;
}