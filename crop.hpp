#ifndef CROP_HPP
#define CROP_HPP

#include <string.h>
#include <algorithm>
#include <vector>
#include <time.h>
#include <cmath>
#include "page_dewarp.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#define PAGE_MARGIN_X 8 // reduced px to ignore near L/R edge
#define PAGE_MARGIN_Y 8 //reduced px to ignore near T/B edge

#define ADAPTIVE_WINSZ 55 // window size for adaptive threshold in reduced px

#define TEXT_MIN_WIDTH 15     // min reduced px width of detected text contour
#define TEXT_MIN_HEIGHT 5     // min reduced px height of detected text contour
#define TEXT_MIN_ASPECT 1.5   // filter out text contours below this w/h ratio
#define TEXT_MAX_THICKNESS 10 // max reduced px thickness of detected text contour

#define EDGE_MAX_OVERLAP 1.0  // max reduced px horiz. overlap of contours in span
#define EDGE_MAX_LENGTH 100.0 // max reduced px length of edge connecting contours
#define EDGE_ANGLE_COST 10.0  // cost of angles in edges (tradeoff vs. length)
#define EDGE_MAX_ANGLE 7.5    // maximum change in angle allowed between contours

#define slice(src, a, b) src.begin() + a, src.begin() + b - 1

#define SPAN_MIN_WIDTH 40   // minimum reduced px width for span
#define SPAN_PX_PER_STEP_CROP 25 // reduced px spacing for sampling along spans

#define MASK_TYPE_TEXT 0
#define MASK_TYPE_LINE 1

using namespace std;
using namespace cv;

void pre_resize_to_screen(cv::Mat src, 
					cv::Mat *dst, 
					double *back_scl, 
					int max_width, 
					int max_height);

void pre_blob_mean_and_tangent (vector<cv::Point> contour, double *center, double *tangent);

void pre_get_mask (string name, 
				cv::Mat small, 
				cv::Mat page_mask, 
				int mask_type,
				cv::Mat *mask); 

void pre_make_tight_mask(std::vector<cv::Point> contour, 
					int min_x, 
					int min_y, 
					int width, 
					int height, 
					cv::Mat *tight_mask);

void pre_get_contours_s(string name, 
					cv::Mat small, 
					cv::Mat page_mask, 
					int mask_type,
					vector<ContourInfo> &contours_out);

double pre_angle_dist(double angle_b, double angle_a);

void pre_generate_candidate_edge(ContourInfo *cinfo_a, 
							ContourInfo *cinfo_b, 
							Edge *var_Edge);

void pre_assemble_spans(string name,
					cv::Mat small,
					cv::Mat page_mask,
					vector<ContourInfo> cinfo_list,
					vector<vector<ContourInfo>> *spans);

vector<cv::Point2d> pre_pix2norm(cv::Size s, vector<cv::Point2d> pts);

vector<cv::Point2d> pre_norm2pix(cv::Size s, std::vector<cv::Point2d> pts, bool as_integer);

void pre_sample_spans (cv::Size shape,
					vector<vector<ContourInfo>> spans,
					vector<vector<cv::Point2d>> *spans_points);

void pre_text_mask (Mat small, 
				int *xmin, 
				int *xmax, 
				int *ymin, 
				int *ymax, 
				int height, 
				int width, 
				vector<vector<cv::Point2d>> *span_points);

void pre_get_page_extents (cv::Mat small, 
					   cv::Mat *page_mask, 
					   std::vector<cv::Point> *page_outline,
					   vector<vector<cv::Point2d>> *span_points);

void pre_keypoints_from_samples(cv::Mat small,
							cv::Mat page_mask,
							vector<cv::Point> page_outline,
							vector<vector<cv::Point2d>> span_points,
							vector<cv::Point2d> *corners,
							vector<vector<double>> *xcoords,
							vector<double> *ycoords, 
							vector<cv::Point2f> *rect);

void pre_crop_image_with_4_point(cv::Mat img, 
							vector<cv::Point2f> outline, 
							double back_scl, 
							cv::Mat *crop_img);

void pre_find_contour_text_area(cv::Mat img, 
							vector<cv::Point2f> *four_corner_text_area,
							double *scl);    

void pre_crop_book_page(Mat input, Mat &output);                 

#endif