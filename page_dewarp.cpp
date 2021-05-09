#include <cmath>
#include <iostream>
#include <string.h>
#include <algorithm>
#include <vector>

#include "page_dewarp.hpp"
#include "crop.hpp"
#include "nlopt.hpp"

#include <time.h>


using namespace std;
using namespace cv;

vector<int> keypoint_index[2];
vector<double> out_params;
vector<cv::Point2d> span_points_flat;
vector<cv::Point2d> corners;
double dims[2];		//giá trị chiều rộng và chiều cao của trang sách

void resize_to_screen(cv::Mat src, 
					cv::Mat *dst, 
					int max_width = 1280, 
					int max_height = 700) 
{
    int width = src.size().width;
    int height = src.size().height;

    double scale_x = double(width) /max_width;
    double scale_y = double(height) / max_height;

    int scale = (int) ceil(scale_x > scale_y ? scale_x : scale_y);

	if (scale > 1) {
		double invert_scale = 1.0 / (double)scale;
		cv::resize(src, *dst, cv::Size(0, 0), invert_scale, invert_scale, INTER_AREA);
	}
	else {
		*dst = src.clone();
	}    
}

void blob_mean_and_tangent (vector<cv::Point> contour, double *center, double *tangent) {
    //Hàm này tìm ra trọng tâm và vector chỉ phương của contour sử dụng SVDecomp
    cv::Moments moments = cv::moments(contour);
    double area = moments.m00;
    double mean_x = moments.m10 / area;
    double mean_y = moments.m01 / area;

    cv::Mat moments_matrix (2, 2, CV_64F);
    moments_matrix.at<double>(0, 0) = moments.mu20 / area;
    moments_matrix.at<double>(0, 1) = moments.mu11 / area;
	moments_matrix.at<double>(1, 0) = moments.mu11 / area;
	moments_matrix.at<double>(1, 1) = moments.mu02 / area;

    cv::Mat svd_u, svd_w, svd_vt;

	cv::SVDecomp(moments_matrix, svd_w, svd_u, svd_vt);
	center[0] = mean_x;
	center[1] = mean_y;

	tangent[0] = svd_u.at<double>(0, 0);
	tangent[1] = svd_u.at<double>(1, 0);
}

void get_mask (string name, 
				cv::Mat small, 
				cv::Mat page_mask, 
				int mask_type,
				cv::Mat *mask) 
{
	cv::Mat sgray;
	cv::cvtColor(small, sgray, cv::COLOR_RGB2GRAY);

	cv::Mat element;

	if(mask_type == MASK_TYPE_TEXT) {
		cv::adaptiveThreshold(sgray, *mask, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, ADAPTIVE_WINSZ, 25);

		element = cv::getStructuringElement(MORPH_RECT, cv::Size(5, 1));
		cv::dilate(*mask, *mask, element);

		element = cv::getStructuringElement(MORPH_RECT, cv::Size(1, 3));
		cv::erode(*mask, *mask, element);
	}
	
	if(mask_type == MASK_TYPE_LINE) {
		cv::adaptiveThreshold(sgray, *mask, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, ADAPTIVE_WINSZ, 7);
		element = cv::getStructuringElement(MORPH_RECT, cv::Size(3, 1));
		cv::erode(*mask, *mask, element, cv::Point(-1, -1), 3);
		
		element = cv::getStructuringElement(MORPH_RECT, cv::Size(8, 2));
		cv::dilate(*mask, *mask, element);
		
	}

	for(unsigned int i = 0; i < mask->rows; ++i) {
		for(unsigned int j = 0; j < mask->cols; ++j) {
			if ((int)mask->at<unsigned char>(i, j) >(int)page_mask.at<unsigned char>(i, j))
				mask->at<unsigned char>(i, j) = page_mask.at<unsigned char>(i, j);

			else continue;
		}
	}
}

void make_tight_mask(std::vector<cv::Point> contour, 
					int min_x, 
					int min_y, 
					int width, 
					int height, 
					cv::Mat *tight_mask) 
{
	// each mask is a zeroes matrix whose width and height are equal to the width and height
	// of the bounding rect of each contour

	*tight_mask = cv::Mat::zeros(height, width, CV_8UC1);
	vector<cv::Point> tight_contour(contour.size());

	for(unsigned int i = 0; i < tight_contour.size(); ++i) {
		tight_contour[i].x = contour[i].x - min_x;
		tight_contour[i].y = contour[i].y - min_y;
	}

	// the tight contour is the original contour remove to the upper left corner
	// to fit into the tight_mask

	vector<vector<cv::Point>> tight_contours(1, tight_contour);
	cv::drawContours(*tight_mask, tight_contours, 0, Scalar(1, 1, 1), -1);

}

void get_contours_s(string name, 
					cv::Mat small, 
					cv::Mat page_mask, 
					int mask_type,
					vector<ContourInfo> &contours_out) 
{
	cv::Mat mask, hierarchy;
	get_mask(name, small, page_mask, mask_type, &mask);

	vector<vector<cv::Point>> contours;
	// vector<vector<cv::Point>> contours_debug;	//cái này chỉ để vẽ contour ra minh họa, sau này đưa lên android sẽ xóa đi

	cv::findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	cv::Rect rect;
	int min_x, min_y, width, height;
	cv::Mat tight_mask;

	for(unsigned int i = 0; i < contours.size(); ++i) {
		rect = cv::boundingRect(contours[i]);
		min_x = rect.x;
		min_y = rect.y;
		width = rect.width;
		height = rect.height;

		if (width < TEXT_MIN_WIDTH || height < TEXT_MIN_HEIGHT || width < TEXT_MIN_ASPECT * height)
			continue;

		make_tight_mask(contours[i], min_x, min_y, width, height, &tight_mask);

		cv::Mat reduced_tight_mask;

		cv::reduce(tight_mask, reduced_tight_mask, 0, cv::REDUCE_SUM, CV_32SC1);	//chuyển matrix thành vector với mỗi phần tử là tổng
																					//các cột

		double min, max;
		cv::minMaxLoc(reduced_tight_mask, &min, &max);

		if (max > TEXT_MAX_THICKNESS) {
			continue;
		}

		contours_out.push_back(ContourInfo(contours[i], rect, tight_mask));
		//contours_debug.push_back(contours[i]);		//cái này chỉ để vẽ contour ra minh họa, sau này đưa lên android sẽ xóa đi
	}

}

double angle_dist(double angle_b, double angle_a)
{
	double diff = angle_b - angle_a;

	while (diff > M_PI)
	{
		diff -= 2 * M_PI;
	}
	while (diff < -M_PI)
	{
		diff += 2 * M_PI;
	}
	return diff > 0 ? diff : -diff;
}

void generate_candidate_edge(ContourInfo *cinfo_a, 
							ContourInfo *cinfo_b, 
							Edge *var_Edge)
{
	bool swap = false;
	ContourInfo _cinfo_a(*cinfo_a);
	ContourInfo _cinfo_b(*cinfo_b);

	if (_cinfo_a.point0[0] > _cinfo_b.point1[0])
	{
		swap = true;
		ContourInfo temp(_cinfo_a);
		_cinfo_a = _cinfo_b;
		_cinfo_b = temp;
	}

	double x_overlap_a = _cinfo_a.local_overlap(_cinfo_b);
	double x_overlap_b = _cinfo_b.local_overlap(_cinfo_a);


	double overall_tangent[2];
	overall_tangent[0] = _cinfo_b.center[0] - _cinfo_a.center[0];
	overall_tangent[1] = _cinfo_b.center[1] - _cinfo_a.center[1];

	double overall_angle = atan2(overall_tangent[1], overall_tangent[0]);

	double delta_angle = std::max(
		angle_dist(_cinfo_a.angle, overall_angle),
		angle_dist(_cinfo_b.angle, overall_angle)) *
		180 / M_PI;

	double x_overlap = std::max(x_overlap_a, x_overlap_b);
	double dist = sqrt(
		pow(_cinfo_b.point0[0] - _cinfo_a.point1[0], 2) +
		pow(_cinfo_b.point0[1] - _cinfo_a.point1[1], 2));

	if (dist > EDGE_MAX_LENGTH || x_overlap > EDGE_MAX_OVERLAP || delta_angle > EDGE_MAX_ANGLE)
	{
		return;
	}
	else
	{
		double score = dist + delta_angle * EDGE_ANGLE_COST;
		if (swap)
		{
			*var_Edge = Edge(score, cinfo_b, cinfo_a);
		}
		else
		{
			*var_Edge = Edge(score, cinfo_a, cinfo_b);
		}
	}
}

void assemble_spans(string name,
					cv::Mat small,
					cv::Mat page_mask,
					vector<ContourInfo> cinfo_list,
					vector<vector<ContourInfo>> *spans)
{
	//sử dụng thuật toán sắp xếp insertion sort sắp xếp các contour theo thứ tự tăng dần
	for (int i = 0; i < cinfo_list.size(); ++i)
	{
		ContourInfo x = cinfo_list[i];		//lưu ý, cần tối ưu lại dòng này
		int j = i;
		while (j > 0 && cinfo_list[j - 1].rect.y > x.rect.y)
		{
			cinfo_list[j] = cinfo_list[j - 1];
			j--;
		}
		cinfo_list[j] = x;
	}

	vector<Edge> candidate_edges;

	for (unsigned int i = 0; i < cinfo_list.size(); ++i)
	{
		for (unsigned int j = 0; j < i; ++j)
		{
			Edge edge;		//cần tối ưu lại dòng này
			generate_candidate_edge(&cinfo_list[i], &cinfo_list[j], &edge);

			if (edge.score)
			{
				candidate_edges.push_back(edge);
			}
		}
	}

	for (int i = 0; i < candidate_edges.size(); ++i)
	{
		Edge x = candidate_edges[i];	//cần tối ưu lại dòng này
		int j = i;
		while (j > 0 && candidate_edges[j - 1].score > x.score)
		{
			candidate_edges[j] = candidate_edges[j - 1];
			j--;
		}
		candidate_edges[j] = x;
	}

	for (unsigned int i = 0; i < candidate_edges.size(); ++i)
	{
		if (candidate_edges[i].cinfo_a->succ == NULL && candidate_edges[i].cinfo_b->pred == NULL)
		{
			candidate_edges[i].cinfo_a->succ = candidate_edges[i].cinfo_b;
			candidate_edges[i].cinfo_b->pred = candidate_edges[i].cinfo_a;
		}
	}

	for (unsigned int i = 0; i < cinfo_list.size(); ++i)
	{
		if (cinfo_list[i].pred != NULL)
		{
			continue;
		}
		ContourInfo *ci = &cinfo_list[i];
		std::vector<ContourInfo> cur_span;
		double width = 0;
		while (ci)
		{
			cur_span.push_back(*ci);
			width += ci->local_xrng[1] - ci->local_xrng[0];
			ci = ci->succ;
		}
		if (width > SPAN_MIN_WIDTH)
		{
			spans->push_back(cur_span);
		}
	}
}

vector<cv::Point2d> pix2norm(cv::Size s, vector<cv::Point2d> pts) {
	std::vector<cv::Point2d> pts_out(pts.size());
	double height = s.height;
	double width = s.width;

	double scl = 2.0 / std::max(width, height);
	cv::Point2d offset(width * 0.5, height * 0.5);
	for (int i = 0; i < pts.size(); ++i)
	{
		pts_out[i].x = (pts[i].x - offset.x) * scl;
		pts_out[i].y = (pts[i].y - offset.y) * scl;
	}
	return pts_out;
}

std::vector<cv::Point2d> norm2pix(cv::Size s, std::vector<cv::Point2d> pts, bool as_integer)
{
	double height = s.height;
	double width = s.width;
	unsigned int i;
	std::vector<cv::Point2d> pts_out(pts.size());
	double scl = std::max(width, height) * 0.5;
	cv::Point offset(width * 0.5, height * 0.5);
	for (i = 0; i < pts.size(); ++i)
	{
		pts_out[i].x = pts[i].x * scl + offset.x;
		pts_out[i].y = pts[i].y * scl + offset.y;
		if (as_integer)
		{
			pts[i].x = (double)(pts[i].x + 0.5);
			pts[i].y = (double)(pts[i].y + 0.5);
		}
	}
	return pts_out;
}

void sample_spans (cv::Size shape,
					vector<vector<ContourInfo>> spans,
					vector<vector<cv::Point2d>> *spans_points) 
{
	for(unsigned int i = 0; i < spans.size(); ++i) {
		vector<cv::Point2d> contour_points;

		for(unsigned int j = 0; j < spans[i].size(); ++j) {
			cv::Mat mask = cv::Mat(spans[i][j].mask);		//xem lại cần tối ưu 2 dòng này
			cv::Rect rect = cv::Rect(spans[i][j].rect);

			int height = mask.size().height;
			int width = mask.size().width;

			vector<int> yvals(height);

			for(unsigned int k = 0; k < height; ++k) {
				yvals[k] = k;
			}

			vector<int> totals(width, 0);
			for(unsigned int c = 0; c < width; ++c) {
				for(unsigned int r = 0; r < height; ++r) {
					totals[c] += (int)mask.at<uchar>(r, c) * yvals[r];
				}
			}

			vector<int> mask_sum(width, 0);
			for (unsigned int k = 0; k < mask_sum.size(); ++k)
			{
				for (int l = 0; l < mask.size().height; ++l)
				{
					mask_sum[k] += (int)mask.at<uchar>(l, k);
				}
			}

			std::vector<double> means(width);
			for (unsigned int k = 0; k < mask_sum.size(); ++k)
			{
				means[k] = (double)totals[k] / (double)mask_sum[k];
			}
			int min_x = rect.x;
			int min_y = rect.y;

			int start = ((width - 1) % SPAN_PX_PER_STEP) / 2;	//điểm bắt đầu của một point trong 1 span. Công thức này có là vì: ví dụ 3 đoạn thẳng nối lại với nhau thì cần 
																//có 4 điểm => số đoạn thẳng + 1 = số điểm cần vẽ. Mà số đoạn thẳng thì bằng chiều dài của đoạn thẳng (coi height là chiều dài đoạn cần chia)
																//chia cho chiều dài của mỗi đoạn (coi per_step là chiều dài mỗi đoạn)

			for (int x = start; x < width; x += SPAN_PX_PER_STEP)
			{
				contour_points.push_back(cv::Point2d((double)x + (double)min_x, means[x] + (double)min_y));
			}
		}
		contour_points = pix2norm(shape, contour_points);
		spans_points->push_back(contour_points);
	}
}

void text_mask (Mat small, 
				int *xmin, 
				int *xmax, 
				int *ymin, 
				int *ymax, 
				int height, 
				int width
				) 
{
	cv::Mat pagemask = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
	cv::rectangle(pagemask, Point(0, 0), Point(width, height), Scalar(255, 255, 255), -1);
	
	vector<ContourInfo> cinfo_list;
	get_contours_s("contour_text_mask", small, pagemask, 0, cinfo_list);
	vector<vector<ContourInfo>> spans_text, spans2;
	assemble_spans("spans_text_maks", small, pagemask, cinfo_list, &spans_text);

	if (spans_text.size() < 3) {
		get_contours_s("contour_text_mask", small, pagemask, 1, cinfo_list);
		assemble_spans("spans_text_mask", small, pagemask, cinfo_list, &spans2);
		if (spans2.size() > spans_text.size()) {
			spans_text = spans2;
		}
	}

	vector<int> tmp_width, tmp_height;
	vector<int> span_width_min;
	vector<int> span_width_max;
	vector<int> span_height_min;
	vector<int> span_height_max;

	for (unsigned int i = 0; i < spans_text.size(); ++i) {
		for (unsigned int j = 0; j < spans_text[i].size(); ++j) {
			for (unsigned int k = 0; k < spans_text[i][j].contour.size(); ++k) {
				tmp_width.push_back(spans_text[i][j].contour[k].x);
				tmp_height.push_back(spans_text[i][j].contour[k].y);
			}
			int maxElementIndex_width = std::max_element(tmp_width.begin(), tmp_width.end()) - tmp_width.begin();
			int minElementIndex_width = std::min_element(tmp_width.begin(), tmp_width.end()) - tmp_width.begin();

			int maxElementIndex_height = std::max_element(tmp_height.begin(), tmp_height.end()) - tmp_height.begin();
			int minElementIndex_height = std::min_element(tmp_height.begin(), tmp_height.end()) - tmp_height.begin();

			span_width_min.push_back(spans_text[i][j].contour[minElementIndex_width].x);
			span_width_max.push_back(spans_text[i][j].contour[maxElementIndex_width].x);

			span_height_min.push_back(spans_text[i][j].contour[minElementIndex_height].y);
			span_height_max.push_back(spans_text[i][j].contour[maxElementIndex_height].y);

			tmp_width.clear();
			tmp_height.clear();
		}
	}

	*xmin = *min_element(span_width_min.begin(), span_width_min.end());
	*xmax = *max_element(span_width_max.begin(), span_width_max.end());
	*ymin = *min_element(span_height_min.begin(), span_height_min.end());
	*ymax = *max_element(span_height_max.begin(), span_height_max.end());
}

void get_page_extents (cv::Mat small, 
					   cv::Mat *page_mask, 
					   std::vector<cv::Point> *page_outline
					   )
{
	int width = small.size().width;
	int height = small.size().height;

	int xmin, xmax, ymin, ymax;

	text_mask(small, &xmin, &xmax, &ymin, &ymax, height, width);
	xmin = xmin - PAGE_MARGIN_X;
	ymin = ymin - PAGE_MARGIN_Y;
	xmax = xmax + PAGE_MARGIN_X;
	ymax = ymax + PAGE_MARGIN_Y;

	*page_mask = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);

	cv::rectangle(*page_mask, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 255, 255), -1);

	page_outline->push_back(cv::Point(xmin, ymin));
	page_outline->push_back(cv::Point(xmin, ymax));
	page_outline->push_back(cv::Point(xmax, ymax));
	page_outline->push_back(cv::Point(xmax, ymin));
}


void keypoints_from_samples(cv::Mat small,
							cv::Mat page_mask,
							vector<cv::Point> page_outline,
							vector<vector<cv::Point2d>> span_points,
							vector<cv::Point2d> *corners,
							vector<vector<double>> *xcoords,
							vector<double> *ycoords
							)
{
	cv::Point2d all_evecs(0.0, 0.0);
	double all_weights = 0;

	for(unsigned int i = 0; i < span_points.size(); ++i) {
		cv::Mat data_pts(span_points[i].size(), 2, CV_64FC1);

		for(unsigned int j = 0; j < data_pts.size().height; ++j) {
			data_pts.at<double>(j, 0) = span_points[i][j].x;
			data_pts.at<double>(j, 1) = span_points[i][j].y;
		}

		// Perform PCA: giảm chiều dữ liệu
		/*
			Mục đích của giảm chiều dữ liệu: cắt bớt chiều dữ liệu, cắt giảm tính toán, giữ lại thông tin có tính quan trọng cao,
			nhằm tăng tốc độ xử lý nhưng vẫn giữ lại được thông tin nhiều nhất có thể
		*/

		cv::PCA pca(data_pts, cv::Mat(), CV_PCA_DATA_AS_ROW, 10);

		cv::Point2d _evec(
				pca.eigenvectors.at<double>(0, 0),
				pca.eigenvectors.at<double>(0, 1));

		double weight = sqrt(
				pow(span_points[i].back().x - span_points[i][0].x, 2) +
				pow(span_points[i].back().y - span_points[i][0].y, 2));

		all_evecs.x += _evec.x * weight;
		all_evecs.y += _evec.y * weight;

		all_weights += weight;
	}

	cv::Point2d evec = all_evecs / all_weights;

	cv::Point2d x_dir(evec);

	if(x_dir.x < 0) x_dir = -x_dir;

	cv::Point2d y_dir(-x_dir.y, x_dir.x);

	std::vector<cv::Point> _pagecoords;

	cv::convexHull(page_outline, _pagecoords);		//vẽ đường outline

	vector<cv::Point2d> pagecoords(_pagecoords.size());
	for(unsigned int i = 0; i < pagecoords.size(); ++i) {
		pagecoords[i].x = (double)_pagecoords[i].x;
		pagecoords[i].y = (double)_pagecoords[i].y;
	}

	pagecoords = pix2norm(page_mask.size(), pagecoords);

	std::vector<double> px_coords(pagecoords.size());
	std::vector<double> py_coords(pagecoords.size());

	for (unsigned int i = 0; i < pagecoords.size(); ++i)
	{
		px_coords[i] = pagecoords[i].x * x_dir.x + pagecoords[i].y * x_dir.y;
		py_coords[i] = pagecoords[i].x * y_dir.x + pagecoords[i].y * y_dir.y;
	}

	double px0, px1, py0, py1;

	cv::minMaxLoc(px_coords, &px0, &px1);
	cv::minMaxLoc(py_coords, &py0, &py1);

	cv::Point2d p00 = px0 * x_dir + py0 * y_dir;
	cv::Point2d p10 = px1 * x_dir + py0 * y_dir;
	cv::Point2d p11 = px1 * x_dir + py1 * y_dir;
	cv::Point2d p01 = px0 * x_dir + py1 * y_dir;

	corners->push_back(p00);
	corners->push_back(p10);
	corners->push_back(p11);
	corners->push_back(p01);

	for(unsigned int i = 0; i < span_points.size(); ++i) {
		std::vector<double> _px_coords(span_points[i].size());
		std::vector<double> _py_coords(span_points[i].size());

		for (unsigned int j = 0; j < span_points[i].size(); ++j) {
			_px_coords[j] = span_points[i][j].x * x_dir.x + span_points[i][j].y * x_dir.y;
			_py_coords[j] = span_points[i][j].x * y_dir.x + span_points[i][j].y * y_dir.y;
		}

		double _py_coords_mean = 0; 
		for (unsigned int k = 0; k < _py_coords.size(); ++k)
		{
			_py_coords_mean += _py_coords[k];
			_px_coords[k] -= px0;
		}
		_py_coords_mean /= _py_coords.size();
		xcoords->push_back(_px_coords);			//xcoords là một điểm (point) trên mỗi spans (thể hiện trong ảnh spans_point.png)
		ycoords->push_back(_py_coords_mean - py0);
	}
}

void get_default_params(vector<cv::Point2d> corners,
						vector<double> ycoords,
						vector<vector<double>> xcoords,
						double *rough_dims,
						vector<int> *span_counts,
						vector<double> *params)
{
	double page_width = sqrt(
		pow(corners[1].x - corners[0].x, 2) +
		pow(corners[1].y - corners[0].y, 2));
	double page_height = sqrt(
		pow(corners[corners.size() - 1].x - corners[0].x, 2) +
		pow(corners[corners.size() - 1].y - corners[0].y, 2));

	rough_dims[0] = page_width;			//xem lại 2 dòng này, có cần cấp phát bộ nhớ ko???
	rough_dims[1] = page_height;

	vector<cv::Point3d> corners_object3d;

	//ban đầu coi trang sách là phẳng nên trục z = 0
	corners_object3d.push_back(cv::Point3d(0, 0, 0));
	corners_object3d.push_back(cv::Point3d(page_width, 0, 0));
	corners_object3d.push_back(cv::Point3d(page_width, page_height, 0));
	corners_object3d.push_back(cv::Point3d(0, page_height, 0));

	cv::Mat rvec; 
	cv::Mat tvec; 

	cv::Mat cameraMatrix(3, 3, cv::DataType<double>::type);
	cameraMatrix = 0;
	cameraMatrix.at<double>(0, 0) = FOCAL_LENGTH;
	cameraMatrix.at<double>(1, 1) = FOCAL_LENGTH;
	cameraMatrix.at<double>(2, 2) = 1;

	//xác định vector xoay rvec và vector dịch tvec của máy ảnh
	cv::solvePnP(corners_object3d, corners, cameraMatrix, cv::Mat::zeros(5, 1, CV_64F), rvec, tvec);

	params->push_back(rvec.at<double>(0, 0));
	params->push_back(rvec.at<double>(1, 0));
	params->push_back(rvec.at<double>(2, 0));
	params->push_back(tvec.at<double>(0, 0));
	params->push_back(tvec.at<double>(1, 0));
	params->push_back(tvec.at<double>(2, 0));

	params->push_back(0);
	params->push_back(0);

	params->insert(params->end(), ycoords.begin(), ycoords.end()); 

	for(unsigned int i = 0; i < xcoords.size(); ++i) {
		span_counts->push_back(xcoords[i].size());	//spans_count là tổng số point được chấm trên một spans 
		params->insert(params->end(), xcoords[i].begin(), xcoords[i].end());
	}
}

void Optimize::make_keypoint_index(vector<int> span_counts) {
	int nspans = (int) span_counts.size();	//tổng số spans

	int npts = 0;	//tổng tất cả các point tính trên tất cả các span
	int start;

	for(unsigned int i = 0; i < span_counts.size(); ++i) {
		npts += span_counts[i];
	}

	keypoint_index[0] = vector<int>(npts + 1, 0);
	keypoint_index[1] = vector<int>(npts + 1, 0);

	start = 1;

	for(unsigned int i = 0; i < span_counts.size(); ++i) {
		for(unsigned int j = start; j < start + span_counts[i]; ++j) {
			keypoint_index[1][j] = 8 + i;
		}

		start = start + span_counts[i];
	}

	for(unsigned int i = 0; i < npts; ++i) {
		keypoint_index[0][i] = i - 1 + 8 + nspans;
	}
}

void polyval(vector<double> poly, 
			vector<cv::Point2d> xy_coords, 
			vector<cv::Point3d> *objpoints)
{
	double x2, x3, z;
	for (unsigned int i = 0; i < xy_coords.size(); i++)
	{
		//printf
		x2 = xy_coords[i].x * xy_coords[i].x;
		x3 = x2 * xy_coords[i].x;
		z = x3 * poly[0] + x2 * poly[1] + xy_coords[i].x * poly[2] + poly[3];
		objpoints->push_back(cv::Point3d((double)xy_coords[i].x, (double)xy_coords[i].y, (double)z));		//tọa độ của điểm gồm 3 chiều x, y, z
	}
}

void project_xy(vector<cv::Point2d> &xy_coords, 
				vector<double> pvec, 
				vector<cv::Point2d> *imagepoints)
{
	double alpha, beta;

	vector<double> rvec = vector<double>(pvec.begin(), pvec.begin() + 3);
	vector<double> tvec = vector<double>(pvec.begin() + 3, pvec.begin() + 6);

	cv::Mat K(3, 3, cv::DataType<double>::type);
	K = 0;
	K.at<double>(0, 0) = FOCAL_LENGTH;
	K.at<double>(1, 1) = FOCAL_LENGTH;
	K.at<double>(2, 2) = 1;

	cv::Mat distCoeffs(5, 1, cv::DataType<double>::type);	//hệ số biến dạng, giả sử hệ số biến dạng bằng 0
	distCoeffs.at<double>(0) = 0;
	distCoeffs.at<double>(1) = 0;
	distCoeffs.at<double>(2) = 0;
	distCoeffs.at<double>(3) = 0;
	distCoeffs.at<double>(4) = 0;

	alpha = pvec[6];
	beta = pvec[7];

	std::vector<double> poly;	//các hệ số của phương trình bậc 3
	std::vector<cv::Point3d> objpoints; 

	//(α+β)x³-(2α+β)x²+αx=0
	poly.push_back(alpha + beta);
	poly.push_back(-2 * alpha - beta);
	poly.push_back(alpha);
	poly.push_back(0);
	polyval(poly, xy_coords, &objpoints);		//tìm tọa độ 3D

	//đưa tọa độ 3D vào hàm cv::projectPoints để tìm ra tọa độ điểm 2D trên mặt phẳng ảnh

	cv::projectPoints(objpoints, rvec, tvec, K, distCoeffs, *imagepoints);	// tính toán hình chiếu các điểm 3D lên mặt phẳng ảnh với các thông số camera trong và ngoài
}

//tìm hình chiếu đống keypoints
void project_keypoints(const vector<double> &pvec, 
						vector<int> *keypoint_index, 
						vector<cv::Point2d> *imagepoints) 
{
	vector<cv::Point2d> xy_coords(keypoint_index[0].size());

	xy_coords[0].x = 0;
	xy_coords[0].y = 0;

	for(unsigned int i = 1; i < keypoint_index[0].size(); ++i) {
		xy_coords[i].x = pvec[keypoint_index[0][i]];
		xy_coords[i].y = pvec[keypoint_index[1][i]];
	}

	project_xy(xy_coords, pvec, imagepoints);
}

double be_like_target(const std::vector<double> &x, 
					std::vector<double> &grad, 
					void *my_func_data)
{
	/*Giá trị trả về phải là giá trị của hàm tại điểm x, 
	trong đó x là một vectơ có độ dài n của các tham số tối ưu hóa
	Tham số độ dốc grad và my_func_data không sử dụng đến
	*/

	vector<cv::Point2d> imagepoints;
	double min = 0;
	project_keypoints(x, keypoint_index, &imagepoints);

	for (unsigned int i = 0; i < imagepoints.size() - 1; i++)
	{
		min += pow(cv::norm(span_points_flat[i] - imagepoints[i]), 2);
	}

	return min;
}

double Optimize::Minimize(vector<double> params) {
	//optimize params

	nlopt::opt opt(nlopt::LN_PRAXIS, params.size());	//n là số lượng tham số tối ưu hóa
	vector<double> lb(params.size(), -5.0);
	vector<double> lu(params.size(), 5.0);

	//Các ràng buộc giới hạn (Bound constraints), đặt giới hạn trên, giới hạn dưới cho hàm tối ưu hóa
	//lb, lu là các vector có độ dài n, cùng với độ dài đc truyền vào constructor opt. (Ở đây là params.size())

	opt.set_lower_bounds(lb);
	opt.set_upper_bounds(lu);
	opt.set_min_objective(be_like_target, NULL);		//chọn chức năng tối thiểu hóa sử dụng hàm set_min_objective, trong đó be_like_target phải có dạng như trên
	
	//Đặt dung sai tuyệt đối cho giá trị hàm
	opt.set_xtol_rel(1e-3);

	//Stop when the number of function evaluations exceeds maxeval.
	opt.set_maxtime(1000);	// có thể xem xét giảm xuống maxtime xuống để tăng tốc độ nhưng output đầu ra sẽ không đẹp bằng
	double opt_f;
	opt.optimize(params, opt_f);
	//khi optimize thành công, x sẽ là các giá trị đã được tối ưu hóa

	for(unsigned int i = 0; i < params.size(); ++i) {
		out_params.push_back(params[i]);
	}

	return 0;
}

int round_nearest_multiple(double i, int factor)
{
	int _i = (int)i;
	int rem = _i % factor; 
	if (!rem)
	{
		return _i;
	}
	else
	{
		return _i + factor - rem;
	}
}

void Optimize::remap_image(string name, 
						cv::Mat img, 
						cv::Mat small, 
						cv::Mat &thresh, 
						vector<double> page_dims, 
						vector<double> params, 
						string outfile_prefix)
{
	// page_dims[1] là chiều cao của ảnh được ánh xạ lại
	double height = page_dims[1] / 2 * OUTPUT_ZOOM * img.size().height;		//nhân chiều cao của ảnh sau khi ánh xạ và chiều cao của ảnh gốc, sau đó chia cho 2

	int _height = round_nearest_multiple(height, REMAP_DECIMATE);
	int _width = round_nearest_multiple((double)_height * page_dims[0] / page_dims[1], REMAP_DECIMATE);

	int height_small = _height / REMAP_DECIMATE;	//khoảng cách các điểm theo chiều cao
	int width_small = _width / REMAP_DECIMATE;		//khoảng cách các điểm theo chiều rộng

	int currentIndex;
	std::vector<double> page_x_range(width_small, 0);
	std::vector<double> page_y_range(height_small, 0);
	double x_range_space = page_dims[0] / (width_small - 1);	//khoảng cách các điểm theo chiều cao biểu diễn trên tọa độ
	double y_range_space = page_dims[1] / (height_small - 1);	//khoảng cách các điểm theo chiều rộng biểu diễn trên tọa độ

	for (int i = 1; i < width_small; i++)
	{
		page_x_range[i] = page_x_range[i - 1] + x_range_space;
	}
	for (int j = 1; j < height_small; j++)
	{
		page_y_range[j] = page_y_range[j - 1] + y_range_space;
	}

	std::vector<cv::Point2d> page_xy_coords(width_small * height_small);		//tọa độ x,y của point... vector page_xy_coords là vector toa độ phục vụ cho việc ánh xạ lại
	// x_coord
	for (int j = 0; j < height_small; j++)
	{
		// y_coord
		for (int i = 0; i < width_small; i++)
		{
			currentIndex = j * width_small + i;
			page_xy_coords[currentIndex].x = page_x_range[i];
			page_xy_coords[currentIndex].y = page_y_range[j];
		}
	}

	std::vector<cv::Point2d> image_points;

	project_xy(page_xy_coords, params, &image_points);

	image_points = norm2pix(img.size(), image_points, false);		// thay đổi nó thành tọa độ thực thông qua hàm Norm2pix

	cv::Mat image_x_coords(cv::Size(width_small, height_small), CV_32FC1);
	cv::Mat image_y_coords(cv::Size(width_small, height_small), CV_32FC1);

	for (int j = 0; j < image_x_coords.rows; j++)
	{
		for (int i = 0; i < image_x_coords.cols; i++)
		{
			currentIndex = j * width_small + i;
			image_x_coords.at<float>(j, i) = (float)image_points[currentIndex].x;
			image_y_coords.at<float>(j, i) = (float)image_points[currentIndex].y;
		}
	}

	cv::Mat image_x_out, image_y_out;

	//Để ánh xạ lại tọa độ của ảnh gốc, chuyển đổi tọa độ thông qua cv2.resize.
	cv::resize(image_x_coords, image_x_out, cv::Size(_width, _height), 0, 0, cv::INTER_CUBIC);
	cv::resize(image_y_coords, image_y_out, cv::Size(_width, _height), 0, 0, cv::INTER_CUBIC);
	// cv::Mat img_gray;
	// cv::cvtColor(img, img_gray, cv::COLOR_RGB2GRAY);
	// cv::Mat remapped;
	cv::remap(img, thresh, image_x_out, image_y_out, cv::INTER_CUBIC, cv::BORDER_REPLICATE);

	//cv::adaptiveThreshold(remapped, thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, ADAPTIVE_WINSZ, 20);
}

void find_contour_text_area(cv::Mat img, 
							vector<cv::Point2f> *four_corner_text_area,
							double *scl) 
{
	cv::Mat small, sgray, mask;

	pre_resize_to_screen(img, &small, scl, 1280, 700);
	cv::cvtColor(small, sgray, cv::COLOR_RGB2GRAY);
	cv::adaptiveThreshold(sgray, mask, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, ADAPTIVE_WINSZ, 25);

	cv::Mat element;
	element = cv::getStructuringElement(MORPH_RECT, cv::Size(1, 75));
	cv::dilate(mask, mask, element);

	cv::Mat hierarchy;
	vector<vector<cv::Point>> contours;

	cv::findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	int index = 0;
	double max_area = 0;
	max_area = cv::contourArea(contours[0]);

	double area = 0;

	for(unsigned int i = 0; i < contours.size(); ++i) {
		area = cv::contourArea(contours[i]);
		if(area > max_area) {
			max_area = area;
			index = i;
		}
	}

	cv::Rect rect;
	rect = cv::boundingRect(contours[index]);

	int xmin, ymin, width, height;
	xmin = rect.x;
	ymin = rect.y;
	width = rect.width;
	height = rect.height;

	four_corner_text_area->push_back(cv::Point2f(xmin - 20, ymin));
	four_corner_text_area->push_back(cv::Point2f((xmin + width + 15), ymin));
	four_corner_text_area->push_back(cv::Point2f((xmin + width + 15), (ymin + height)));
	four_corner_text_area->push_back(cv::Point2f(xmin - 20, (ymin + height)));
}



int main() {
    cv::Mat image, image_crop, page_mask, small, img_dst, final_result;
    vector<cv::Point> page_outline;
    vector<ContourInfo> contours_out;
    vector<vector<ContourInfo>> spans, spans2;
    vector<vector<cv::Point2d>> span_points;
    String outfile_prefix;

	clock_t start = clock();

    String filename = "/home/quangnh98/Downloads/test/20.jpg";
	image = cv::imread(filename);

	/*PHẦN 1: TIỀN XỬ LÝ, CROP ẢNH.
	  Mục đích: Loại bỏ tối đa phần dư thừa, 
	  tập trung tối đa vào vùng text cần thiết*/

	  Mat image_crop1;

	pre_crop_book_page(image, image_crop1);
	pre_crop_book_page(image_crop1, image_crop);

	cv::imwrite("dcmm.jpg", image_crop);
	
	//***********************************************************************************************************
	
	/* PHẦN 2: DEWARP ẢNH VỪA CROP */

    resize_to_screen(image_crop, &small);

    get_page_extents(small, &page_mask, &page_outline);

	get_contours_s(filename, small, page_mask, 0, contours_out);

	assemble_spans(filename, small, page_mask, contours_out, &spans);

	if (spans.size() < 3) {
		get_contours_s(filename, small, page_mask, 1, contours_out);
		assemble_spans(filename, small, page_mask, contours_out, &spans2);
		if (spans2.size() > spans.size()) {
			spans = spans2;
		}
	}

	if (spans.size() < 1) return 0;

	sample_spans(small.size(), spans, &span_points);

	vector<vector<double>> xcoords;
	std::vector<double> ycoords;

	keypoints_from_samples(small, page_mask, page_outline, span_points, &corners, &xcoords, &ycoords);
	
	double rough_dims[2];
	vector<int> span_counts;
	//params là vector chứa các tham số alpha và beta đặc trưng cho độ cong của trang sách
	vector<double> params;		
	get_default_params(corners, ycoords, xcoords, rough_dims, &span_counts, &params);

	span_points_flat.push_back(corners[0]);

	for (unsigned int i = 0; i < span_points.size(); ++i)
	{
		span_points_flat.insert(span_points_flat.end(), span_points[i].begin(), span_points[i].end());
	}

	Optimize Object;
	Object.make_keypoint_index(span_counts);

	dims[0] = rough_dims[0];
	dims[1] = rough_dims[1];

	Object.Minimize(params);

	vector<double> dim_vector{ dims[0], dims[1] };

	Object.remap_image("output", image_crop, small, img_dst, dim_vector, out_params, outfile_prefix);

	//***********************************************************************************************************
	
	/* PHẦN 3: TIẾP TỤC CROP ẢNH */

	Mat final_img, mat_binary;
	vector<cv::Point2f> vector_4_point;
	double scl;
	find_contour_text_area(img_dst, &vector_4_point, &scl);
	pre_crop_image_with_4_point(img_dst, vector_4_point, scl, &final_img);
	cv::cvtColor(final_img, final_img, cv::COLOR_RGB2GRAY);
	cv::adaptiveThreshold(final_img, mat_binary, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, ADAPTIVE_WINSZ, 20);

	if (DEBUG_LEVEL)
		cv::imwrite("result_output.png", mat_binary);

	printf("\nTime: %.2fs\n", (double)(clock() - start)/CLOCKS_PER_SEC);

    return 0;
}
