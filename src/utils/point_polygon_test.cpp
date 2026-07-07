#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "../global_vars.hpp"
#include "point_polygon_test.hpp"

namespace detect_utils
{

    /**
     * @brief box 多边形测试
     * @param box YOLO 检测框
     * @param polygon 顶点集合
     * @param filter_location 筛选位置
     * @param measureDist 是否返回距离
     * @return 拓扑位置 (-1: 在外部，0: 在边界，1: 在内部)
     */
    double box_polygon_test(
        const Global::YoloDetectBox &box,
        const std::vector<cv::Point> &polygon,
        const FilterLocation filter_location,
        bool measureDist)
    {
        cv::Point2f point;

        if (filter_location == FilterLocation::Center)
        {
            point = {(box.left + box.right) / 2.0f,
                     (box.top + box.bottom) / 2.0f};
        }
        else if (filter_location == FilterLocation::LeftCenter)
        {
            point = {box.left * 1.0f,
                     (box.top + box.bottom) / 2.0f};
        }
        else if (filter_location == FilterLocation::RightCenter)
        {
            point = {box.right * 1.0f,
                     (box.top + box.bottom) / 2.0f};
        }
        else if (filter_location == FilterLocation::TopCenter)
        {
            point = {(box.left + box.right) / 2.0f,
                     box.top * 1.0f};
        }
        else if (filter_location == FilterLocation::BottomCenter)
        {
            point = {(box.left + box.right) / 2.0f,
                     box.bottom * 1.0f};
        }
        else if (filter_location == FilterLocation::LeftTop)
        {
            point = {box.left * 1.0f,
                     box.top * 1.0f};
        }
        else if (filter_location == FilterLocation::RightTop)
        {
            point = {box.right * 1.0f,
                     box.top * 1.0f};
        }
        else if (filter_location == FilterLocation::LeftBottom)
        {
            point = {box.left * 1.0f,
                     box.bottom * 1.0f};
        }
        else if (filter_location == FilterLocation::RightBottom)
        {
            point = {box.right * 1.0f,
                     box.bottom * 1.0f};
        }

        // 调用 OpenCV 的点多边形测试函数
        // 参数 measureDist = false: 只返回拓扑位置（1 在内部，0 在边界，-1 在外部）
        double result = cv::pointPolygonTest(polygon, point, measureDist);
        return result;
    }

    /**
     * @brief 计算两个点集的 IOU
     * @param points1 第一个点集
     * @param points2 第二个点集
     * @param use_ioa 是否使用 IOA (bool，默认为 false), IOA 指的是交的面积占 points2 的面积
     * @return IOU 或 IOA 值
     */
    double calc_polygons_iou(
        const std::vector<cv::Point> &points1,
        const std::vector<cv::Point> &points2,
        const bool use_ioa)
    {
        // 至少需要 3 个点才能构成有效多边形
        if (points1.size() < 3 || points2.size() < 3)
        {
            return 0.0;
        }

        // 连续几何面积为 0 的情况，直接返回 0
        // 避免一堆共线点被 fillPoly 填成奇怪的像素线段面积
        const double area1 = std::abs(cv::contourArea(points1));
        const double area2 = std::abs(cv::contourArea(points2));

        if (area1 <= 1e-6 || area2 <= 1e-6)
        {
            return 0.0;
        }

        // 计算两个点集共同的外接区域，避免创建整张大图
        cv::Rect rect1 = cv::boundingRect(points1);
        cv::Rect rect2 = cv::boundingRect(points2);
        cv::Rect roi = rect1 | rect2;

        if (roi.width <= 0 || roi.height <= 0)
        {
            return 0.0;
        }

        // 留一点边距，避免边界点刚好卡在 mask 边缘
        constexpr int padding = 2;

        cv::Size mask_size(
            roi.width + padding * 2,
            roi.height + padding * 2);

        auto shift_points = [&](const std::vector<cv::Point> &points)
        {
            std::vector<cv::Point> shifted;
            shifted.reserve(points.size());

            for (const auto &p : points)
            {
                shifted.emplace_back(
                    p.x - roi.x + padding,
                    p.y - roi.y + padding);
            }

            return shifted;
        };

        std::vector<cv::Point> shifted1 = shift_points(points1);
        std::vector<cv::Point> shifted2 = shift_points(points2);

        cv::Mat mask1(mask_size, CV_8UC1, cv::Scalar(0));
        cv::Mat mask2(mask_size, CV_8UC1, cv::Scalar(0));

        std::vector<std::vector<cv::Point>> poly1{shifted1};
        std::vector<std::vector<cv::Point>> poly2{shifted2};

        cv::fillPoly(mask1, poly1, cv::Scalar(255), cv::LINE_8);
        cv::fillPoly(mask2, poly2, cv::Scalar(255), cv::LINE_8);

        cv::Mat inter_mask;
        cv::bitwise_and(mask1, mask2, inter_mask);

        const double inter_area = static_cast<double>(cv::countNonZero(inter_mask));

        double denom = 0.0;

        if (use_ioa)
        {
            // IOA = intersection / area(points2)
            denom = static_cast<double>(cv::countNonZero(mask2));
        }
        else
        {
            // IOU = intersection / union
            cv::Mat union_mask;
            cv::bitwise_or(mask1, mask2, union_mask);
            denom = static_cast<double>(cv::countNonZero(union_mask));
        }

        if (denom <= 1e-6)
        {
            return 0.0;
        }

        return inter_area / denom;
    }

    /**
     * @brief 在多边形区域内过滤 YOLO 检测框
     * @param boxes YOLO 检测框的集合 (std::vector<Global::YoloDetectBox>)
     * @param polygon 表示多边形顶点的集合 (std::vector<cv::Point>)
     * @param inside 是否只保留多边形内部的检测框 (bool，默认为 true)
     * @param filter_location 筛选位置 (FilterLocation，默认为 FilterLocation::Center)
     * @return 在多边形内的检测框 index 集合 (std::vector<int>)
     */
    std::vector<int> filter_box_ids_by_polygon(
        const std::vector<Global::YoloDetectBox> &boxes,
        const std::vector<cv::Point> &polygon,
        const bool inside,
        const FilterLocation filter_location)
    {
        if (boxes.empty())
            return {};

        // 如果多边形顶点少于 3 个，无法构成有效区域，保持原逻辑：视为全部在内部
        if (polygon.size() < 3)
        {
            std::vector<int> index_list;
            index_list.reserve(boxes.size());
            for (int i = 0; i < boxes.size(); i++)
            {
                index_list.push_back(i);
            }
            return index_list; // 直接返回原列表的 id
        }

        std::vector<int> inside_ids;
        inside_ids.reserve(boxes.size());

        for (int i = 0; i < boxes.size(); i++)
        {
            auto &box = boxes[i];

            auto result = box_polygon_test(box, polygon, filter_location, false);

            // 包含在内部 (result > 0) 或正好在边界上 (result == 0)
            if (inside && result >= 0)
            {
                inside_ids.push_back(i);
            }
            else if (!inside && result <= 0)
            {
                inside_ids.push_back(i);
            }
        }

        return inside_ids;
    }

    /**
     * @brief 在多边形区域内过滤 YOLO 检测框
     * @param boxes YOLO 检测框的集合 (std::vector<Global::YoloDetectBox>)
     * @param polygon 表示多边形顶点的集合 (std::vector<cv::Point>)
     * @param inside 是否只保留多边形内部的检测框 (bool，默认为 true)
     * @param filter_location 筛选位置 (FilterLocation，默认为 FilterLocation::Center)
     * @return 在多边形内的检测框集合 (std::vector<Global::YoloDetectBox>)
     */
    std::vector<Global::YoloDetectBox> filter_boxes_by_polygon(
        const std::vector<Global::YoloDetectBox> &boxes,
        const std::vector<cv::Point> &polygon,
        const bool inside,
        const FilterLocation filter_location)
    {
        if (boxes.empty())
            return {};

        // 如果多边形顶点少于 3 个，无法构成有效区域，保持原逻辑：视为全部在内部
        if (polygon.size() < 3)
        {
            return boxes; // 直接返回原列表的拷贝
        }

        std::vector<Global::YoloDetectBox> inside_boxes;
        inside_boxes.reserve(boxes.size());

        for (const auto &box : boxes)
        {
            auto result = box_polygon_test(box, polygon, filter_location, false);

            // 包含在内部 (result > 0) 或正好在边界上 (result == 0)
            if (inside && result >= 0)
            {
                inside_boxes.push_back(box);
            }
            else if (!inside && result <= 0)
            {
                inside_boxes.push_back(box);
            }
        }

        return inside_boxes;
    }

    /**
     * @brief 在图像上绘制封闭的多边形线条
     * @param image 要在其上绘制的图像 (cv::Mat)
     * @param polygon_points 多边形的顶点集合 (std::vector<cv::Point>)
     * @param color 线条颜色 (cv::Scalar，例如 cv::Scalar(0, 255, 0) 为绿色)
     * @param thickness 线条粗细 (int，默认为 2)
     */
    void draw_closed_polygon(
        cv::Mat &image,
        const std::vector<cv::Point> &polygon_points,
        const cv::Scalar &color,
        const int thickness)
    {
        if (polygon_points.size() < 2)
            return;

        std::vector<std::vector<cv::Point>> polylines_data;
        polylines_data.push_back(polygon_points);

        cv::polylines(image, polylines_data, true, color, thickness, cv::LINE_AA);
    }

    void test_filter_boxes_by_polygon()
    {
        std::cout << "===================== test_filter_boxes_by_polygon =====================" << std::endl;

        // 1. 创建一张空白的黑色图像用于测试 (高600, 宽800)
        cv::Mat test_img = cv::Mat::zeros(600, 800, CV_8UC3);

        // 2. 定义一个凹多边形的顶点
        std::vector<cv::Point> my_polygon;
        // x, y
        my_polygon.push_back(cv::Point(100, 100)); // 顶点 1
        my_polygon.push_back(cv::Point(400, 50));  // 顶点 2
        my_polygon.push_back(cv::Point(700, 150)); // 顶点 3
        my_polygon.push_back(cv::Point(500, 400)); // 顶点 4
        my_polygon.push_back(cv::Point(400, 250)); // 顶点 5 (造成凹陷的点)
        my_polygon.push_back(cv::Point(200, 500)); // 顶点 6

        // 3. 模拟 YOLO 检测结果
        std::vector<Global::YoloDetectBox> boxes;
        // 框1：中心点 (300, 200) -> 应该在多边形内
        boxes.push_back({0, "person", 0.9f, 280, 150, 320, 250});
        // 框2：中心点 (100, 400) -> 应该在多边形外 (左下角空白处)
        boxes.push_back({0, "person", 0.8f, 80, 350, 120, 450});
        // 框3：中心点 (400, 350) -> 应该在多边形外 (刚好掉进那个凹陷区域)
        boxes.push_back({1, "bicycle", 0.95f, 350, 300, 450, 400});
        // 框4：中心点 (500, 150) -> 应该在多边形内
        boxes.push_back({2, "car", 0.85f, 450, 100, 550, 200});
        // 框5：中心点 (100, 100) -> 应该在多边形上 (左上角)
        boxes.push_back({3, "motorcycle", 0.7f, 40, 50, 160, 150});
        // 框6：中心点 (700, 300) -> 应该在多边形外 (右下角空白处)
        boxes.push_back({4, "bus", 0.6f, 650, 250, 750, 350});

        // 4. 执行检测算法，获取在区域内的检测框
        std::vector<int> inside_ids = filter_box_ids_by_polygon(boxes, my_polygon);

        // 5. 可视化绘制
        // 5.1 绘制多边形边框 (黄色)
        cv::Scalar poly_color(0, 255, 255);
        draw_closed_polygon(test_img, my_polygon, poly_color, 2);

        // 5.2 遍历所有原始检测框并绘制
        for (int i = 0; i < boxes.size(); i++)
        {
            auto &box = boxes[i];
            cv::Point center(
                (box.left + box.right) / 2,
                (box.top + box.bottom) / 2);

            // 通过匹配坐标来判断当前框是否在过滤后的结果中
            bool is_inside = false;
            if (std::find(inside_ids.begin(), inside_ids.end(), i) != inside_ids.end())
            {
                is_inside = true;
            }

            // 设定颜色：内部为绿色，外部为红色 (BGR格式)
            cv::Scalar box_color = is_inside ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

            cv::rectangle(test_img, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), box_color, 2, cv::LINE_AA);
            cv::circle(test_img, center, 4, box_color, -1, cv::LINE_AA);
            cv::putText(test_img, box.class_name, cv::Point(box.left, box.top - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, box_color, 1, cv::LINE_AA);
        }

        // 6. 保存与显示结果
        cv::imwrite("test_filter_boxes_by_polygon.jpg", test_img);
        std::cout << "Saved test_filter_boxes_by_polygon.jpg" << std::endl;

        cv::imshow("YOLO Polygon Detection", test_img);
        cv::waitKey(0);

        std::cout << "===================== test_filter_boxes_by_polygon =====================" << std::endl
                  << std::endl;
    }

    void test_calc_polygons_iou()
    {
        std::cout << "===================== test_calc_polygons_iou =====================" << std::endl;

        // 1. 创建一张空白的黑色图像用于测试 (高600, 宽800)
        cv::Mat test_img = cv::Mat::zeros(600, 800, CV_8UC3);

        // 2. 定义两个多边形的顶点
        std::vector<cv::Point> points1 = {cv::Point(100, 100), cv::Point(300, 100), cv::Point(300, 300), cv::Point(100, 300)};
        std::vector<cv::Point> points2 = {cv::Point(200, 200), cv::Point(400, 200), cv::Point(400, 400), cv::Point(200, 400)};

        // 3. 计算两个多边形的 IOU
        // 第1组
        double iou = calc_polygons_iou(points1, points2);
        double ioa = calc_polygons_iou(points1, points2, true);
        std::cout << "IOU: " << iou << std::endl;
        std::cout << "IOA: " << ioa << std::endl;

        // 黄色
        cv::Scalar poly_color1(0, 255, 255);
        // 粉色
        cv::Scalar poly_color2(255, 0, 255);
        draw_closed_polygon(test_img, points1, poly_color1, 2);
        draw_closed_polygon(test_img, points2, poly_color2, 2);

        std::string classString = "IOU: " + std::to_string(iou) + ", IOA: " + std::to_string(ioa);
        cv::putText(test_img, classString, cv::Point(500, 500), cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1, cv::LINE_AA);

        cv::imwrite("test_calc_polygons_iou1.jpg", test_img);
        std::cout << "Saved test_calc_polygons_iou1.jpg" << std::endl;

        cv::imshow("test_calc_polygons_iou", test_img);
        cv::waitKey(0);

        // 第2组
        test_img = cv::Mat::zeros(600, 800, CV_8UC3);
        points1 = {cv::Point(200, 0), cv::Point(400, 0), cv::Point(400, 500), cv::Point(200, 500)};
        points2 = {cv::Point(100, 200), cv::Point(300, 200), cv::Point(300, 100), cv::Point(500, 100), cv::Point(500, 300), cv::Point(300, 300), cv::Point(300, 400), cv::Point(100, 400)};
        iou = calc_polygons_iou(points1, points2);
        ioa = calc_polygons_iou(points1, points2, true);
        std::cout << "IOU: " << iou << std::endl;
        std::cout << "IOA: " << ioa << std::endl;

        draw_closed_polygon(test_img, points1, poly_color1, 2);
        draw_closed_polygon(test_img, points2, poly_color2, 2);

        classString = "IOU: " + std::to_string(iou) + ", IOA: " + std::to_string(ioa);
        cv::putText(test_img, classString, cv::Point(500, 500), cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1, cv::LINE_AA);

        cv::imwrite("test_calc_polygons_iou2.jpg", test_img);
        std::cout << "Saved test_calc_polygons_iou2.jpg" << std::endl;

        cv::imshow("test_calc_polygons_iou", test_img);
        cv::waitKey(0);

        // 第3组
        test_img = cv::Mat::zeros(600, 800, CV_8UC3);
        points1 = {cv::Point(100, 100), cv::Point(300, 100), cv::Point(100, 300)};
        points2 = {cv::Point(300, 100), cv::Point(300, 300), cv::Point(100, 300)};
        iou = calc_polygons_iou(points1, points2);
        ioa = calc_polygons_iou(points1, points2, true);
        std::cout << "IOU: " << iou << std::endl;
        std::cout << "IOA: " << ioa << std::endl;

        draw_closed_polygon(test_img, points1, poly_color1, 2);
        draw_closed_polygon(test_img, points2, poly_color2, 2);

        classString = "IOU: " + std::to_string(iou) + ", IOA: " + std::to_string(ioa);
        cv::putText(test_img, classString, cv::Point(500, 500), cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1, cv::LINE_AA);

        cv::imwrite("test_calc_polygons_iou3.jpg", test_img);
        std::cout << "Saved test_calc_polygons_iou3.jpg" << std::endl;

        cv::imshow("test_calc_polygons_iou", test_img);
        cv::waitKey(0);

        // 第4组
        test_img = cv::Mat::zeros(600, 800, CV_8UC3);
        points1 = {cv::Point(100, 100), cv::Point(300, 100), cv::Point(100, 300)};
        points2 = {cv::Point(400, 100), cv::Point(400, 300), cv::Point(200, 300)};
        iou = calc_polygons_iou(points1, points2);
        ioa = calc_polygons_iou(points1, points2, true);
        std::cout << "IOU: " << iou << std::endl;
        std::cout << "IOA: " << ioa << std::endl;

        draw_closed_polygon(test_img, points1, poly_color1, 2);
        draw_closed_polygon(test_img, points2, poly_color2, 2);

        classString = "IOU: " + std::to_string(iou) + ", IOA: " + std::to_string(ioa);
        cv::putText(test_img, classString, cv::Point(500, 500), cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1, cv::LINE_AA);

        cv::imwrite("test_calc_polygons_iou4.jpg", test_img);
        std::cout << "Saved test_calc_polygons_iou4.jpg" << std::endl;

        cv::imshow("test_calc_polygons_iou", test_img);
        cv::waitKey(0);

        std::cout << "===================== test_calc_polygons_iou =====================" << std::endl
                  << std::endl;
    }

}