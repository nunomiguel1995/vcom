#include "pch.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/**  @function main */
int main(int argc, char** argv)
{

	vector<vector<Point>> contours, cropped_contours;
	Mat src, hsv, hsv_final, ycrcb, ycrcb_final, bgr_final, gray, binary, cropped_image;
	Rect roi;

	/// Load image
	src = imread("C:\\Users\\Joao\\source\\repos\\openCvTry\\openCvTry\\quina.jpg");

	cvtColor(src, hsv, COLOR_BGR2HSV);

	vector<Mat> split_hsv;
	split(hsv, split_hsv);

	split_hsv[2] = 255;

	merge(split_hsv, hsv_final);

	Mat temp;

	cvtColor(hsv_final, temp, COLOR_HSV2BGR);

	for (int i = 1; i < 15; i = i + 2)
	{
		GaussianBlur(temp, temp, Size(i, i), 0, 0);
	}

	cvtColor(temp, ycrcb, COLOR_BGR2YCrCb);

	vector<Mat> spl; split(ycrcb, spl);

	spl[1] = 0;
	spl[2] = 0;

	merge(spl, ycrcb_final);
	cvtColor(ycrcb_final, bgr_final, COLOR_YCrCb2BGR);
	
	Mat test_hsv;
	cvtColor(bgr_final, gray, COLOR_RGB2GRAY);
	threshold(gray, binary, 100, 255, THRESH_BINARY_INV | THRESH_OTSU);

	imshow("binary", binary);

	vector<Vec4i> hierarchy;

	findContours(binary, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	int min_x = 99999, min_y = 99999, max_x = 0, max_y = 0;
	int max_contour = 0;
	int max_contour_index = NULL;

	if (contours.size() > 1) {
		for (size_t i = 1; i < contours.size(); i++) {
			contours[i].insert(contours[0].end(), contours[i].begin(), contours[i].end());
		}
	}

	for (size_t j = 0; j < contours[0].size(); j++) {
		Point p = contours[0][j];
		if (p.x < min_x)
			min_x = p.x;
		if (p.x > max_x)
			max_x = p.x;
		if (p.y < min_y)
			min_y = p.y;
		if (p.y > max_y)
			max_y = p.y;
	}

	roi = Rect(min_x, min_y, max_x - min_x, max_y - min_y);

	cropped_image = binary(roi);

	findContours(cropped_image, cropped_contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	vector<Moments> mu(cropped_contours.size());
	for (int i = 0; i < cropped_contours.size(); i++)
	{
		mu[i] = moments(cropped_contours[i], false);
	}

	///  Get the mass centers:
	vector<Point2f> mc(cropped_contours.size());
	for (int i = 0; i < cropped_contours.size(); i++)
	{
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

	Mat cropped_img_rgb;
	cvtColor(cropped_image, cropped_img_rgb, COLOR_GRAY2BGR);

	circle(cropped_img_rgb, (Point) mc[0], 4, Scalar(127, 127, 127), 2);

	imshow("centroid", cropped_img_rgb);
	/// Wait until user exits the program*/
	waitKey(0);

	return 0;
}