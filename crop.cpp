#include "crop.hpp"

void pre_resize_to_screen(cv::Mat src, 
					cv::Mat *dst, 
					double *back_scl, 
					int max_width, 
					int max_height) 
{
    int width = src.size().width;
    int height = src.size().height;

    double scale_x = double(width) /max_width;
    double scale_y = double(height) / max_height;

    int scale = (int) ceil(scale_x > scale_y ? scale_x : scale_y);

	if (scale > 1) {
		double invert_scale = 1.0 / (double)scale;
		*back_scl = scale;
		cv::resize(src, *dst, cv::Size(0, 0), invert_scale, invert_scale, INTER_AREA);
	}
	else {
		*back_scl = 1;
		*dst = src.clone();
	}    
}

void pre_blob_mean_and_tangent (vector<cv::Point> contour, double *center, double *tangent) {
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

double ContourInfo::project_x (cv::Point point) {
    return (this->tangent[0] * (point.x - this->center[0]) + this->tangent[1] * (point.y - this->center[1]));
}

double ContourInfo::interval_measure_overlap (double *int_a, double *int_b) {
    return std::min(int_a[1], int_b[1]) - std::max(int_a[0], int_b[0]);
}

double ContourInfo::local_overlap (ContourInfo other) {
	double int_b[2];
	int_b[0] = project_x(cv::Point(other.point0[0], other.point0[1]));
	int_b[1] = project_x(cv::Point(other.point1[0], other.point1[1]));  

    return interval_measure_overlap(this->local_xrng, int_b);  
}

ContourInfo::ContourInfo() {
    point0[0] = 0;
	point0[1] = 0;
	point1[0] = 0;
	point1[1] = 0;
	center[0] = 0;
	center[1] = 0;
	angle = 0;
	pred = NULL;
	succ = NULL;
	local_xrng[0] = 0;
	local_xrng[1] = 0;
	mask = 0;
	tangent[0] = 0;
	tangent[1] = 0;	
}

ContourInfo::ContourInfo(const ContourInfo &c) {
    point0[0] = c.point0[0];
	point0[1] = c.point0[1];
	point1[0] = c.point1[0];
	point1[1] = c.point1[1];
	center[0] = c.center[0];
	center[1] = c.center[1];
	angle = c.angle;
	pred = c.pred;
	succ = c.succ;
	local_xrng[0] = c.local_xrng[0];
	local_xrng[1] = c.local_xrng[1];
	contour = c.contour;
	rect = c.rect;
	mask = c.mask;
	tangent[0] = c.tangent[0];
	tangent[1] = c.tangent[1];
}

ContourInfo::ContourInfo(vector<cv::Point> c, cv::Rect r, cv::Mat m) {
    this->contour = c;
    this->rect = r;
    this->mask = m;

    //tìm trọng tâm và vector chỉ phương của contour
    pre_blob_mean_and_tangent(this->contour, this->center, this->tangent);
    this->angle = atan2(this->tangent[1], this->tangent[0]);

    double clx[contour.size()];

	for (int i = 0; i < contour.size(); ++i)
	{
		clx[i] = this->project_x(c[i]);
	}

	double min_lx = *std::min_element(clx, clx + contour.size());
	double max_lx = *std::max_element(clx, clx + contour.size());

	this->local_xrng[0] = min_lx;
	this->local_xrng[1] = max_lx;

	this->point0[0] = this->center[0] + this->tangent[0] * min_lx;
	this->point0[1] = this->center[1] + this->tangent[1] * min_lx;
	this->point1[0] = this->center[0] + this->tangent[0] * max_lx;
	this->point1[1] = this->center[1] + this->tangent[1] * max_lx;

	this->pred = NULL;
	this->succ = NULL;
}

bool ContourInfo::operator == (const ContourInfo &ci0) const {
    return (
		point0[0] == ci0.point0[0] &&
		point0[1] == ci0.point0[1] &&
		point1[0] == ci0.point1[0] &&
		point1[1] == ci0.point1[1] &&
		center[0] == ci0.center[0] &&
		center[1] == ci0.center[1] &&
		angle == ci0.angle &&
		local_xrng[0] == ci0.local_xrng[0] &&
		local_xrng[1] == ci0.local_xrng[1] &&
		contour.size() == ci0.contour.size()
	);
}

void ContourInfo::operator = (const ContourInfo &c) {
    point0[0] = c.point0[0];
	point0[1] = c.point0[1];
	point1[0] = c.point1[0];
	point1[1] = c.point1[1];
	center[0] = c.center[0];
	center[1] = c.center[1];
	angle = c.angle;
	pred = c.pred;
	succ = c.succ;
	local_xrng[0] = c.local_xrng[0];
	local_xrng[1] = c.local_xrng[1];
	contour = c.contour;
	rect = c.rect;
	mask = c.mask;
	tangent[0] = c.tangent[0];
	tangent[1] = c.tangent[1];
}

void pre_get_mask (string name, 
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

void pre_make_tight_mask(std::vector<cv::Point> contour, 
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

void pre_get_contours_s(string name, 
					cv::Mat small, 
					cv::Mat page_mask, 
					int mask_type,
					vector<ContourInfo> &contours_out) 
{
	cv::Mat mask, hierarchy;
	pre_get_mask(name, small, page_mask, mask_type, &mask);

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

		pre_make_tight_mask(contours[i], min_x, min_y, width, height, &tight_mask);

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

double pre_angle_dist(double angle_b, double angle_a)
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

void pre_generate_candidate_edge(ContourInfo *cinfo_a, 
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
		pre_angle_dist(_cinfo_a.angle, overall_angle),
		pre_angle_dist(_cinfo_b.angle, overall_angle)) *
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

void pre_assemble_spans(string name,
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
			pre_generate_candidate_edge(&cinfo_list[i], &cinfo_list[j], &edge);

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
		// cout << "DEBUG_ASSEMBLE_SPANS_6:" << cinfo_list[i].pred << "/" << cinfo_list[i].succ << endl;
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

vector<cv::Point2d> pre_pix2norm(cv::Size s, vector<cv::Point2d> pts) {
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

std::vector<cv::Point2d> pre_norm2pix(cv::Size s, std::vector<cv::Point2d> pts, bool as_integer)
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

void pre_sample_spans (cv::Size shape,
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

			int start = ((width - 1) % SPAN_PX_PER_STEP_CROP) / 2;	//điểm bắt đầu của một point trong 1 span. Công thức này có là vì: ví dụ 3 đoạn thẳng nối lại với nhau thì cần 
																//có 4 điểm => số đoạn thẳng + 1 = số điểm cần vẽ. Mà số đoạn thẳng thì bằng chiều dài của đoạn thẳng (coi height là chiều dài đoạn cần chia)
																//chia cho chiều dài của mỗi đoạn (coi per_step là chiều dài mỗi đoạn)

			for (int x = start; x < width; x += SPAN_PX_PER_STEP_CROP)
			{
				contour_points.push_back(cv::Point2d((double)x + (double)min_x, means[x] + (double)min_y));
			}
		}
		contour_points = pre_pix2norm(shape, contour_points);
		spans_points->push_back(contour_points);
	}
}

void pre_text_mask (Mat small, 
				int *xmin, 
				int *xmax, 
				int *ymin, 
				int *ymax, 
				int height, 
				int width, 
				vector<vector<cv::Point2d>> *span_points) 
{
	cv::Mat pagemask = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
	cv::rectangle(pagemask, Point(0, 0), Point(width, height), Scalar(255, 255, 255), -1);
	
	vector<ContourInfo> cinfo_list;
	pre_get_contours_s("contour_text_mask", small, pagemask, 0, cinfo_list);
	vector<vector<ContourInfo>> spans_text, spans2;
	pre_assemble_spans("spans_text_maks", small, pagemask, cinfo_list, &spans_text);

	if (spans_text.size() < 3) {
		pre_get_contours_s("contour_text_mask", small, pagemask, 1, cinfo_list);
		pre_assemble_spans("spans_text_mask", small, pagemask, cinfo_list, &spans2);
		if (spans2.size() > spans_text.size()) {
			spans_text = spans2;
		}
	}

	pre_sample_spans(small.size(), spans_text, span_points);

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

void pre_get_page_extents (cv::Mat small, 
					   cv::Mat *page_mask, 
					   std::vector<cv::Point> *page_outline,
					   vector<vector<cv::Point2d>> *span_points)
{
	int width = small.size().width;
	int height = small.size().height;

	int xmin, xmax, ymin, ymax;

	pre_text_mask(small, &xmin, &xmax, &ymin, &ymax, height, width, span_points);
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


void pre_keypoints_from_samples(cv::Mat small,
							cv::Mat page_mask,
							vector<cv::Point> page_outline,
							vector<vector<cv::Point2d>> span_points,
							vector<cv::Point2d> *corners,
							vector<vector<double>> *xcoords,
							vector<double> *ycoords, 
							vector<cv::Point2f> *rect)
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

	pagecoords = pre_pix2norm(page_mask.size(), pagecoords);

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

	vector<cv::Point2d> rval = pre_norm2pix(small.size(), *corners, true);
	
	rect->push_back(cv::Point2f(rval[0].x - 20, rval[0].y - 20));
	rect->push_back(cv::Point2f(rval[1].x + 20, rval[1].y - 20));
	rect->push_back(cv::Point2f(rval[2].x + 20, rval[2].y + 35));
	rect->push_back(cv::Point2f(rval[3].x - 20, rval[3].y + 35));
}

void pre_crop_image_with_4_point(cv::Mat img, 
							vector<cv::Point2f> outline, 
							double back_scl, 
							cv::Mat *crop_img) 
{
	cv::Point2f rect[4];
	rect[0] = cv::Point2f(outline[0].x * back_scl, outline[0].y * back_scl);
	rect[1] = cv::Point2f(outline[1].x * back_scl, outline[1].y * back_scl);
	rect[2] = cv::Point2f(outline[2].x * back_scl, outline[2].y * back_scl);
	rect[3] = cv::Point2f(outline[3].x * back_scl, outline[3].y * back_scl);

	cv::Point2f tl, tr, br, bl;
	tl = rect[0];
	tr = rect[1];
	br = rect[2];
	bl = rect[3];

	double widthA = sqrt(pow((br.x - bl.x), 2) + pow((br.y - bl.y), 2));
	double widthB = sqrt(pow((tr.x - tl.x), 2) + pow((tr.y - tl.y), 2));
	double maxWidth = std::max((int)widthA, (int)widthB);
			
	double heightA = sqrt(pow((tr.x - br.x), 2) + pow((tr.y - br.y), 2));
	double heightB = sqrt(pow((tl.x - bl.x), 2) + pow((tl.y - bl.y), 2));
	double maxHeight = std::max((int)heightA, (int)heightB);

	cv::Point2f dst[4];
	dst[0] = cv::Point2f(0, 0);
	dst[1] = cv::Point2f(maxWidth - 1, 0);
	dst[2] = cv::Point2f(maxWidth - 1, maxHeight - 1);
	dst[3] = cv::Point2f(0, maxHeight - 1);

	cv::Mat M;
	M = cv::getPerspectiveTransform(rect, dst);			
	cv::warpPerspective(img, *crop_img, M, cv::Size(maxWidth, maxHeight));				
}

void pre_find_contour_text_area(cv::Mat img, 
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

void pre_crop_book_page (Mat input, Mat &output) {
    cv::Mat pre_small, crop_with_outline_img;
	cv::Mat pre_pagemask;
	vector<cv::Point> pre_page_outline;
	double pre_back_scl, scl_text_area;
	vector<vector<cv::Point2d>> pre_span_points;
	vector<cv::Point2d> pre_corners;
	vector<vector<double>> pre_xcoords;
	std::vector<double> pre_ycoords;
	vector<cv::Point2f> pre_rect;
	vector<cv::Point2f> four_corner_text_area;

    pre_resize_to_screen(input, &pre_small, &pre_back_scl, 1280, 700);

	pre_get_page_extents(pre_small, &pre_pagemask, &pre_page_outline, &pre_span_points);
	pre_keypoints_from_samples(pre_small, 
							pre_pagemask,
							pre_page_outline, 
							pre_span_points, 
							&pre_corners, 
							&pre_xcoords, 
							&pre_ycoords,
							&pre_rect);

	pre_crop_image_with_4_point(input, pre_rect, pre_back_scl, &crop_with_outline_img);
	pre_find_contour_text_area(crop_with_outline_img, &four_corner_text_area, &scl_text_area);
	pre_crop_image_with_4_point(crop_with_outline_img, four_corner_text_area, scl_text_area, &output);
}

