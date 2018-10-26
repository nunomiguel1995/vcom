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
	src = imread("C:\\Users\\Joao\\source\\repos\\openCvTry\\openCvTry\\quina.jpg");

	Mat hsv;
	cvtColor(src, hsv, COLOR_BGR2HSV);

	vector<Mat> split_hsv;
	split(hsv, split_hsv);

	split_hsv[2] = 255;

	Mat hsv_final;
	merge(split_hsv, hsv_final);

	Mat ycrcb, temp;

	cvtColor(hsv_final, temp, COLOR_HSV2BGR);

	for (int i = 1; i < 15; i = i + 2)
	{
		GaussianBlur(temp, temp, Size(i, i), 0, 0);
	}
	imshow("after blur", temp);

	cvtColor(temp, ycrcb, COLOR_BGR2YCrCb);

	vector<Mat> spl; split(ycrcb, spl);

	//spl[0] = 0;
	spl[1] = 0;
	spl[2] = 0;

	Mat final; merge(spl, final);
	cvtColor(final, final, COLOR_YCrCb2BGR);
	
	Mat test_hsv;
	cvtColor(final, final, COLOR_RGB2GRAY);
	threshold(final, final, 100, 255, THRESH_BINARY_INV | THRESH_OTSU);

	imshow("binary", final);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(final, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image

	/*double largestArea = 0;
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
	}*/

	int min_x = 99999, min_y = 99999, max_x = 0, max_y = 0;
	int max_contour = 0;
	int max_contour_index = NULL;

	for (size_t i = 0; i < contours.size(); i++) {
		if (contours[i].size() > max_contour) {
			max_contour_index = i;
			min_x = 99999;
			min_y = 99999;
			max_x = 0;
			max_y = 0;
			max_contour = contours[i].size();
			for (size_t j = 0; j < contours[i].size(); j++) {
				Point p = contours[i][j];
				if (p.x < min_x)
					min_x = p.x;
				if (p.x > max_x)
					max_x = p.x;
				if (p.y < min_y)
					min_y = p.y;
				if (p.y > max_y)
					max_y = p.y;
			}
		}
	}

	Rect roi(min_x, min_y, max_x - min_x, max_y - min_y);

	Mat test = final(roi);

	imshow("cropped", test);

	/// Wait until user exits the program*/
	waitKey(0);

	return 0;
}