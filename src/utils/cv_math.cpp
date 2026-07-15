#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include "cv_math.hpp"

namespace detect_utils
{

    /**
     * @brief 计算两个二维点之间的距离
     * @param p1 第一个点
     * @param p2 第二个点
     * @return 返回两点之间的距离（double类型）
     */
    double calc_point_distance(
        const cv::Point2d &p1,
        const cv::Point2d &p2)
    {
        const double dx = p2.x - p1.x;
        const double dy = p2.y - p1.y;

        const double dist = std::hypot(dx, dy);

        // 防止出现 -0.0 或非常接近 0 的浮点误差
        if (std::abs(dist) < eps)
        {
            return 0.0;
        }

        return dist;
    }

    void test_calc_point_distance()
    {
        std::cout << "===================== test_calc_point_distance =====================" << std::endl;
        cv::Point2d a(0.0, 0.0);
        cv::Point2d b(-0.0, -0.0);
        double d = calc_point_distance(a, b);
        std::cout << "a(" << a.x << ", " << a.y << ") and b(" << b.x << ", " << b.y << ")" << " distance: " << d << std::endl;
        // a(0, 0) and b(-0, -0) distance: 0

        a = cv::Point2d(0.0, 0.0);
        b = cv::Point2d(1.0, 0.0);
        d = calc_point_distance(a, b);
        std::cout << "a(" << a.x << ", " << a.y << ") and b(" << b.x << ", " << b.y << ")" << " distance: " << d << std::endl;
        // a(0, 0) and b(1, -0) distance: 1

        a = cv::Point2d(0.0, 0.0);
        b = cv::Point2d(1.0, 1.0);
        d = calc_point_distance(a, b);
        std::cout << "a(" << a.x << ", " << a.y << ") and b(" << b.x << ", " << b.y << ")" << " distance: " << d << std::endl;
        // a(0, 0) and b(1, 1) distance: 1.41421

        std::cout << "===================== test_calc_point_distance =====================" << std::endl
                  << std::endl;
    }

    /**
     * @brief 计算点到线段的距离
     * @param point 要计算距离的点
     * @param segment_start 线段的起点
     * @param segment_end 线段的终点
     * @param projection_mode 投影模式，有 InfiniteLine 和 ClampToSegment 两种
     * @return 返回点到线段的距离（double类型）
     */
    double calc_point_to_segment_distance(
        const cv::Point2d &point,
        const cv::Point2d &segment_start,
        const cv::Point2d &segment_end,
        const ProjectionMode projection_mode)
    {
        const double vx = segment_end.x - segment_start.x;
        const double vy = segment_end.y - segment_start.y;

        const double wx = point.x - segment_start.x;
        const double wy = point.y - segment_start.y;

        const double segment_length_squared = vx * vx + vy * vy;

        // 线段退化为一个点，此时无法定义所在直线
        if (segment_length_squared < eps)
        {
            const double distance = std::hypot(wx, wy);
            return distance < eps ? 0.0 : distance;
        }

        // point 在线段所在直线上的投影参数
        double t = (wx * vx + wy * vy) / segment_length_squared;

        if (projection_mode == ProjectionMode::ClampToSegment)
        {
            t = std::clamp(t, 0.0, 1.0);
        }

        const cv::Point2d projection(
            segment_start.x + t * vx,
            segment_start.y + t * vy);

        const double distance = std::hypot(
            point.x - projection.x,
            point.y - projection.y);

        // 清除接近 0 的误差，同时避免输出 -0.0
        return distance < eps ? 0.0 : distance;
    }

    void test_calc_point_to_segment_distance()
    {
        std::cout << "===================== test_calc_point_to_segment_distance =====================" << std::endl;
        cv::Point2d point(0.0, 0.0);
        cv::Point2d segment_start(0.0, 0.0);
        cv::Point2d segment_end(0.0, 0.0);
        double d = calc_point_to_segment_distance(point, segment_start, segment_end);
        std::cout << "point(" << point.x << ", " << point.y << ") to segment_start(" << segment_start.x << ", " << segment_start.y << ") and segment_end(" << segment_end.x << ", " << segment_end.y << ")"
                  << " distance: " << d << " mode: InfiniteLine" << std::endl;
        d = calc_point_to_segment_distance(point, segment_start, segment_end, ProjectionMode::ClampToSegment);
        std::cout << "point(" << point.x << ", " << point.y << ") to segment_start(" << segment_start.x << ", " << segment_start.y << ") and segment_end(" << segment_end.x << ", " << segment_end.y << ")"
                  << " distance: " << d << " mode: ClampToSegment" << std::endl;
        // point(0, 0) to segment_start(0, 0) and segment_end(0, 0) distance: 0 mode: InfiniteLine
        // point(0, 0) to segment_start(0, 0) and segment_end(0, 0) distance: 0 mode: ClampToSegment

        point = cv::Point2d(0.0, 2.0);
        segment_start = cv::Point2d(0.0, 0.0);
        segment_end = cv::Point2d(1.0, 0.0);
        d = calc_point_to_segment_distance(point, segment_start, segment_end);
        std::cout << "point(" << point.x << ", " << point.y << ") to segment_start(" << segment_start.x << ", " << segment_start.y << ") and segment_end(" << segment_end.x << ", " << segment_end.y << ")"
                  << " distance: " << d << " mode: InfiniteLine" << std::endl;
        d = calc_point_to_segment_distance(point, segment_start, segment_end, ProjectionMode::ClampToSegment);
        std::cout << "point(" << point.x << ", " << point.y << ") to segment_start(" << segment_start.x << ", " << segment_start.y << ") and segment_end(" << segment_end.x << ", " << segment_end.y << ")"
                  << " distance: " << d << " mode: ClampToSegment" << std::endl;
        // point(0, 2) to segment_start(0, 0) and segment_end(1, 0) distance: 2 mode: InfiniteLine
        // point(0, 2) to segment_start(0, 0) and segment_end(1, 0) distance: 2 mode: ClampToSegment

        point = cv::Point2d(0.0, 2.0);
        segment_start = cv::Point2d(1.0, 0.0);
        segment_end = cv::Point2d(2.0, 0.0);
        d = calc_point_to_segment_distance(point, segment_start, segment_end);
        std::cout << "point(" << point.x << ", " << point.y << ") to segment_start(" << segment_start.x << ", " << segment_start.y << ") and segment_end(" << segment_end.x << ", " << segment_end.y << ")"
                  << " distance: " << d << " mode: InfiniteLine" << std::endl;
        d = calc_point_to_segment_distance(point, segment_start, segment_end, ProjectionMode::ClampToSegment);
        std::cout << "point(" << point.x << ", " << point.y << ") to segment_start(" << segment_start.x << ", " << segment_start.y << ") and segment_end(" << segment_end.x << ", " << segment_end.y << ")"
                  << " distance: " << d << " mode: ClampToSegment" << std::endl;
        // point(0, 2) to segment_start(1, 0) and segment_end(2, 0) distance: 2 mode: InfiniteLine
        // point(0, 2) to segment_start(1, 0) and segment_end(2, 0) distance: 2.23607 mode: ClampToSegment

        std::cout << "===================== test_calc_point_to_segment_distance =====================" << std::endl
                  << std::endl;
    }

    /**
     * @brief 这个函数计算的是由三个二维点 a、b、c 构成的两个向量 AB 和 AC 的二维叉乘（Cross Product）结果（即叉积在Z轴方向的标量值）。
     * @param a 第一个点
     * @param b 第二个点
     * @param c 第三个点
     * @return 返回叉积在Z轴方向的标量值（double类型）
     */
    double cross(const cv::Point2d &a, const cv::Point2d &b, const cv::Point2d &c)
    {
        // 向量 AB x AC
        return (b.x - a.x) * (c.y - a.y) -
               (b.y - a.y) * (c.x - a.x);
    }

    /**
     * @brief 判断点 p 是否严格位于由点 ab 上
     * @param a 线段的起点
     * @param b 线段的终点
     * @param p 要判断的点
     * @return 返回点 p 是否在线段 ab 上（bool类型）
     */
    bool on_segment(const cv::Point2d &a, const cv::Point2d &b, const cv::Point2d &p)
    {
        return std::abs(cross(a, b, p)) < eps &&
               p.x >= std::min(a.x, b.x) - eps &&
               p.x <= std::max(a.x, b.x) + eps &&
               p.y >= std::min(a.y, b.y) - eps &&
               p.y <= std::max(a.y, b.y) + eps;
    }

    /**
     * @brief 判断两条线段是否相交
     *        核心思想是：如果两条线段相交，那么其中一条线段的两个端点，必然分布在另一条线段所在直线的两侧。
     * @param p1 线段1的起点
     * @param p2 线段1的终点
     * @param q1 线段2的起点
     * @param q2 线段2的终点
     * @return 返回两条线段是否相交（bool类型）
     */
    bool segments_intersect(
        const cv::Point2d &p1,
        const cv::Point2d &p2,
        const cv::Point2d &q1,
        const cv::Point2d &q2)
    {
        // 这里利用了 cross 函数的“方向判断”特性：
        //   c1 和 c2 代表了点 q1 和 q2 分别位于有向直线 p1p2 的哪一侧（左侧为正，右侧为负）。
        //   c3 和 c4 代表了点 p1 和 p2 分别位于有向直线 q1q2 的哪一侧。
        const double c1 = cross(p1, p2, q1);
        const double c2 = cross(p1, p2, q2);
        const double c3 = cross(q1, q2, p1);
        const double c4 = cross(q1, q2, p2);

        // 一般相交
        // 这是跨立实验的标准形态。
        // 如果 q1 和 q2 在直线 p1p2 的两侧，那么 c1 和 c2 必定符号相反（一正一负）。
        // 因此它们的乘积必定 < 0
        // 同理，必须同时也满足 p1 和 p2 在直线 q1q2 的两侧。
        if ((c1 * c2 < -eps) && (c3 * c4 < -eps))
        {
            return true;
        }

        // 边界条件判定 (非规范相交)
        // 共线或端点接触
        if (std::abs(c1) < eps && on_segment(p1, p2, q1))
            return true;
        if (std::abs(c2) < eps && on_segment(p1, p2, q2))
            return true;
        if (std::abs(c3) < eps && on_segment(q1, q2, p1))
            return true;
        if (std::abs(c4) < eps && on_segment(q1, q2, p2))
            return true;

        return false;
    }

    void test_segments_intersect()
    {
        std::cout << "===================== test_calc_polygons_iou =====================" << std::endl;

        std::vector<cv::Point2d> line1 = {cv::Point2d(200, 600), cv::Point2d(600, 200)};
        std::vector<cv::Point2d> line2 = {cv::Point2d(200, 200), cv::Point2d(600, 600)};
        std::vector<cv::Point2d> line3 = {cv::Point2d(0, 400), cv::Point2d(200, 600)};
        std::vector<cv::Point2d> line4 = {cv::Point2d(400, 200), cv::Point2d(500, 300)};
        std::vector<cv::Point2d> line5 = {cv::Point2d(100, 700), cv::Point2d(400, 400)};
        std::vector<cv::Point2d> line6 = {cv::Point2d(300, 700), cv::Point2d(400, 600)};

        auto line1_2_is_intersect = segments_intersect(line1[0], line1[1], line2[0], line2[1]);
        std::cout << "line1 and line2 is intersect: " << line1_2_is_intersect << std::endl; //

        auto line1_3_is_intersect = segments_intersect(line1[0], line1[1], line3[0], line3[1]);
        std::cout << "line1 and line3 is intersect: " << line1_3_is_intersect << std::endl; //

        auto line1_4_is_intersect = segments_intersect(line1[0], line1[1], line4[0], line4[1]);
        std::cout << "line1 and line4 is intersect: " << line1_4_is_intersect << std::endl; //

        auto line1_5_is_intersect = segments_intersect(line1[0], line1[1], line5[0], line5[1]);
        std::cout << "line1 and line5 is intersect: " << line1_5_is_intersect << std::endl; //

        auto line1_6_is_intersect = segments_intersect(line1[0], line1[1], line6[0], line6[1]);
        std::cout << "line1 and line6 is intersect: " << line1_6_is_intersect << std::endl; //

        cv::Mat test_img = cv::Mat::zeros(800, 1000, CV_8UC3);

        cv::line(test_img, line1[0], line1[1], cv::Scalar(255, 245, 0), 4);
        cv::putText(test_img, "line1", line1[0], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 245, 0), 1, cv::LINE_AA);
        cv::line(test_img, line2[0], line2[1], cv::Scalar(205, 90, 106), 2);
        cv::putText(test_img, "line2", line2[0], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(205, 90, 106), 1, cv::LINE_AA);
        cv::line(test_img, line3[0], line3[1], cv::Scalar(0, 238, 0), 2);
        cv::putText(test_img, "line3", line3[0], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 238, 0), 1, cv::LINE_AA);
        cv::line(test_img, line4[0], line4[1], cv::Scalar(62, 255, 192), 2);
        cv::putText(test_img, "line4", line4[0], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(62, 255, 192), 1, cv::LINE_AA);
        cv::line(test_img, line5[0], line5[1], cv::Scalar(0, 255, 255), 1);
        cv::putText(test_img, "line5", line5[0], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        cv::line(test_img, line6[0], line6[1], cv::Scalar(34, 34, 178), 2);
        cv::putText(test_img, "line6", line6[0], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(34, 34, 178), 1, cv::LINE_AA);

        cv::putText(test_img, "line1 and line2 is intersect: " + std::to_string(line1_2_is_intersect),
                    cv::Point(700, 100), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(205, 116, 24), 1, cv::LINE_AA);
        cv::putText(test_img, "line1 and line3 is intersect: " + std::to_string(line1_3_is_intersect),
                    cv::Point(700, 120), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(205, 116, 24), 1, cv::LINE_AA);
        cv::putText(test_img, "line1 and line4 is intersect: " + std::to_string(line1_4_is_intersect),
                    cv::Point(700, 140), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(205, 116, 24), 1, cv::LINE_AA);
        cv::putText(test_img, "line1 and line5 is intersect: " + std::to_string(line1_5_is_intersect),
                    cv::Point(700, 160), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(205, 116, 24), 1, cv::LINE_AA);
        cv::putText(test_img, "line1 and line6 is intersect: " + std::to_string(line1_6_is_intersect),
                    cv::Point(700, 180), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(205, 116, 24), 1, cv::LINE_AA);

        cv::imwrite("test_segments_intersect.jpg", test_img);
        std::cout << "Saved test_segments_intersect.jpg" << std::endl;

        cv::imshow("test_segments_intersect", test_img);
        cv::waitKey(0);

        std::cout << "===================== test_calc_polygons_iou =====================" << std::endl
                  << std::endl;
    }

    /**
     * @brief 计算两个点之间的角度
     * @param start 起始点
     * @param end 终止点
     * @param coord_system 坐标系类型（OpenCV或Math）
     * @return 返回角度（度）
     */
    double calc_line_angle(
        const cv::Point2d &start,
        const cv::Point2d &end,
        const CoordSystem coord_system)
    {
        const double dx = end.x - start.x;
        double dy = end.y - start.y;

        if (coord_system == CoordSystem::Math)
        {
            dy = -dy;
        }

        // 线段长度为 0，角度无意义，这里统一返回 0
        if (std::abs(dx) < eps && std::abs(dy) < eps)
        {
            return 0.0;
        }

        double angle = std::atan2(dy, dx) * 180.0 / CV_PI;

        // 归一化到 [0, 360)
        angle = std::fmod(angle + 360.0, 360.0);

        // 处理 -0.0、极小误差、接近 360 的情况
        if (std::abs(angle) < eps || std::abs(angle - 360.0) < eps)
        {
            angle = 0.0;
        }

        return angle;
    }

    void test_calc_line_angle()
    {
        std::cout << "===================== test_calc_line_angle =========" << std::endl;

        std::vector<cv::Point2d> line1 = {cv::Point2d(300, 300), cv::Point2d(500, 300)};
        std::vector<cv::Point2d> line2 = {cv::Point2d(300, 300), cv::Point2d(500, 100)};
        std::vector<cv::Point2d> line3 = {cv::Point2d(300, 300), cv::Point2d(300, 100)};
        std::vector<cv::Point2d> line4 = {cv::Point2d(300, 300), cv::Point2d(100, 100)};
        std::vector<cv::Point2d> line5 = {cv::Point2d(300, 300), cv::Point2d(100, 300)};
        std::vector<cv::Point2d> line6 = {cv::Point2d(300, 300), cv::Point2d(100, 500)};
        std::vector<cv::Point2d> line7 = {cv::Point2d(300, 300), cv::Point2d(300, 500)};
        std::vector<cv::Point2d> line8 = {cv::Point2d(300, 300), cv::Point2d(500, 500)};

        auto line1_cv_coord_angle = calc_line_angle(line1[0], line1[1]);
        auto line1_math_coord_angle = calc_line_angle(line1[0], line1[1], CoordSystem::Math);
        std::cout << "line1 cv_coord_angle: " << line1_cv_coord_angle << ", math_coord_angle: " << line1_math_coord_angle << std::endl;

        auto line2_cv_coord_angle = calc_line_angle(line2[0], line2[1]);
        auto line2_math_coord_angle = calc_line_angle(line2[0], line2[1], CoordSystem::Math);
        std::cout << "line2 cv_coord_angle: " << line2_cv_coord_angle << ", math_coord_angle: " << line2_math_coord_angle << std::endl;

        auto line3_cv_coord_angle = calc_line_angle(line3[0], line3[1]);
        auto line3_math_coord_angle = calc_line_angle(line3[0], line3[1], CoordSystem::Math);
        std::cout << "line3 cv_coord_angle: " << line3_cv_coord_angle << ", math_coord_angle: " << line3_math_coord_angle << std::endl;

        auto line4_cv_coord_angle = calc_line_angle(line4[0], line4[1]);
        auto line4_math_coord_angle = calc_line_angle(line4[0], line4[1], CoordSystem::Math);
        std::cout << "line4 cv_coord_angle: " << line4_cv_coord_angle << ", math_coord_angle: " << line4_math_coord_angle << std::endl;

        auto line5_cv_coord_angle = calc_line_angle(line5[0], line5[1]);
        auto line5_math_coord_angle = calc_line_angle(line5[0], line5[1], CoordSystem::Math);
        std::cout << "line5 cv_coord_angle: " << line5_cv_coord_angle << ", math_coord_angle: " << line5_math_coord_angle << std::endl;

        auto line6_cv_coord_angle = calc_line_angle(line6[0], line6[1]);
        auto line6_math_coord_angle = calc_line_angle(line6[0], line6[1], CoordSystem::Math);
        std::cout << "line6 cv_coord_angle: " << line6_cv_coord_angle << ", math_coord_angle: " << line6_math_coord_angle << std::endl;

        auto line7_cv_coord_angle = calc_line_angle(line7[0], line7[1]);
        auto line7_math_coord_angle = calc_line_angle(line7[0], line7[1], CoordSystem::Math);
        std::cout << "line7 cv_coord_angle: " << line7_cv_coord_angle << ", math_coord_angle: " << line7_math_coord_angle << std::endl;

        auto line8_cv_coord_angle = calc_line_angle(line8[0], line8[1]);
        auto line8_math_coord_angle = calc_line_angle(line8[0], line8[1], CoordSystem::Math);
        std::cout << "line8 cv_coord_angle: " << line8_cv_coord_angle << ", math_coord_angle: " << line8_math_coord_angle << std::endl;

        cv::Mat test_img = cv::Mat::zeros(800, 1400, CV_8UC3);

        cv::line(test_img, line1[0], line1[1], cv::Scalar(255, 245, 0), 2);
        cv::putText(test_img, "line1", line1[1], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 245, 0), 1, cv::LINE_AA);
        cv::line(test_img, line2[0], line2[1], cv::Scalar(205, 90, 106), 2);
        cv::putText(test_img, "line2", line2[1], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(205, 90, 106), 1, cv::LINE_AA);
        cv::line(test_img, line3[0], line3[1], cv::Scalar(0, 238, 0), 2);
        cv::putText(test_img, "line3", line3[1], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 238, 0), 1, cv::LINE_AA);
        cv::line(test_img, line4[0], line4[1], cv::Scalar(62, 255, 192), 2);
        cv::putText(test_img, "line4", line4[1], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(62, 255, 192), 1, cv::LINE_AA);
        cv::line(test_img, line5[0], line5[1], cv::Scalar(0, 255, 255), 2);
        cv::putText(test_img, "line5", line5[1], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        cv::line(test_img, line6[0], line6[1], cv::Scalar(34, 34, 178), 2);
        cv::putText(test_img, "line6", line6[1], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(34, 34, 178), 1, cv::LINE_AA);
        cv::line(test_img, line7[0], line7[1], cv::Scalar(255, 245, 0), 2);
        cv::putText(test_img, "line7", line7[1], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(139, 26, 85), 1, cv::LINE_AA);
        cv::line(test_img, line8[0], line8[1], cv::Scalar(205, 90, 106), 2);
        cv::putText(test_img, "line8", line8[1], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);

        cv::putText(test_img, "line1 cv_coord_angle: " + std::to_string(line1_cv_coord_angle) + ", math_coord_angle: " + std::to_string(line1_math_coord_angle),
                    cv::Point(600, 100), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(205, 116, 24), 1, cv::LINE_AA);
        cv::putText(test_img, "line2 cv_coord_angle: " + std::to_string(line2_cv_coord_angle) + ", math_coord_angle: " + std::to_string(line2_math_coord_angle),
                    cv::Point(600, 120), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(205, 116, 24), 1, cv::LINE_AA);
        cv::putText(test_img, "line3 cv_coord_angle: " + std::to_string(line3_cv_coord_angle) + ", math_coord_angle: " + std::to_string(line3_math_coord_angle),
                    cv::Point(600, 140), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(205, 116, 24), 1, cv::LINE_AA);
        cv::putText(test_img, "line4 cv_coord_angle: " + std::to_string(line4_cv_coord_angle) + ", math_coord_angle: " + std::to_string(line4_math_coord_angle),
                    cv::Point(600, 160), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(205, 116, 24), 1, cv::LINE_AA);
        cv::putText(test_img, "line5 cv_coord_angle: " + std::to_string(line5_cv_coord_angle) + ", math_coord_angle: " + std::to_string(line5_math_coord_angle),
                    cv::Point(600, 180), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(205, 116, 24), 1, cv::LINE_AA);
        cv::putText(test_img, "line6 cv_coord_angle: " + std::to_string(line6_cv_coord_angle) + ", math_coord_angle: " + std::to_string(line6_math_coord_angle),
                    cv::Point(600, 200), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(205, 116, 24), 1, cv::LINE_AA);
        cv::putText(test_img, "line7 cv_coord_angle: " + std::to_string(line7_cv_coord_angle) + ", math_coord_angle: " + std::to_string(line7_math_coord_angle),
                    cv::Point(600, 220), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(205, 116, 24), 1, cv::LINE_AA);
        cv::putText(test_img, "line8 cv_coord_angle: " + std::to_string(line8_cv_coord_angle) + ", math_coord_angle: " + std::to_string(line8_math_coord_angle),
                    cv::Point(600, 240), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(205, 116, 24), 1, cv::LINE_AA);

        cv::imwrite("test_calc_line_angle.jpg", test_img);
        std::cout << "Saved test_calc_line_angle.jpg" << std::endl;

        cv::imshow("test_calc_line_angle", test_img);
        cv::waitKey(0);

        std::cout << "===================== test_calc_line_angle =========" << std::endl
                  << std::endl;
    }
}
