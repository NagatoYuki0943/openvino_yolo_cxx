#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "global_vars.hpp"
#include "point_polygon_test.hpp"

namespace point_polygon_test
{

    // 主功能函数
    // 参数 1: boxes - YOLO检测框的集合
    // 参数 2: polygon - 表示多边形顶点的集合 (OpenCV 格式)
    // 返回值: 在多边形内的检测框在原始 vector 中的索引 (Index)
    std::vector<int> filter_boxes_in_polygon(
        const std::vector<Global::YoloDetectBox> &boxes,
        const std::vector<cv::Point> &polygon)
    {
        std::vector<int> inside_indices;

        if (boxes.empty())
            return inside_indices;

        // 如果多边形顶点少于 3 个，无法构成有效区域
        if (polygon.size() < 3)
        {
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                inside_indices.push_back(i);
            }
            return inside_indices;
        }

        for (size_t i = 0; i < boxes.size(); ++i)
        {
            // 计算 YOLO 检测框的中心点
            // 注意：cv::pointPolygonTest 推荐传入 cv::Point2f 类型的点
            cv::Point2f center(
                (boxes[i].left + boxes[i].right) / 2.0f,
                (boxes[i].top + boxes[i].bottom) / 2.0f);

            // 调用 OpenCV 的点多边形测试函数
            // 参数 measureDist = false: 只返回拓扑位置（1 在内部，0 在边界，-1 在外部）
            double result = cv::pointPolygonTest(polygon, center, false);

            // 包含在内部 (result > 0) 或正好在边界上 (result == 0)
            if (result >= 0)
                inside_indices.push_back(i);
        }

        return inside_indices;
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
        // 1. 数据检查：至少需要 2 个点才能画线（虽然 3 个点才能构成封闭平面）
        if (polygon_points.size() < 2)
            return;

        // 2. 准备 cv::polylines 所需的数据结构
        // cv::polylines 接收的是一个“多边形列表”，即 vector<vector<Point>>。
        // 即使我们只画一个多边形，也需要把它包装进一层 vector 中。
        std::vector<std::vector<cv::Point>> polylines_data;
        polylines_data.push_back(polygon_points);

        // 3. 调用函数绘制
        // 参数 1: 目标图像
        // 参数 2: 顶点数组的数组
        // 参数 3: 是否封闭 (true 代表自动收尾相连)
        // 参数 4: 颜色
        // 参数 5: 线条粗细
        // 参数 6: 线条类型 (LINE_AA 代表抗锯齿，线条更平滑)
        cv::polylines(image, polylines_data, true, color, thickness, cv::LINE_AA);
    }

    void test_filter_boxes_in_polygon()
    {
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

        // 4. 执行检测算法，获取在区域内的框的索引
        std::vector<int> inside_indices = filter_boxes_in_polygon(boxes, my_polygon);

        // 5. 可视化绘制
        // 5.1 绘制多边形边框 (黄色)
        cv::Scalar poly_color(0, 255, 255);
        draw_closed_polygon(test_img, my_polygon, poly_color, 3);

        // 5.2 遍历所有检测框并绘制
        for (size_t i = 0; i < boxes.size(); ++i)
        {
            const auto &box = boxes[i];

            // 计算中心点用于画圆
            cv::Point center(
                (box.left + box.right) / 2,
                (box.top + box.bottom) / 2);

            // 判断当前框是否在 inside_indices 列表中
            bool is_inside = std::find(inside_indices.begin(), inside_indices.end(), i) != inside_indices.end();

            // 设定颜色：内部为绿色，外部为红色 (BGR格式)
            cv::Scalar box_color = is_inside ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

            // 绘制矩形框
            cv::rectangle(test_img, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), box_color, 2, cv::LINE_AA);

            // 绘制实心中心点 (半径为4，-1代表实心填充)
            cv::circle(test_img, center, 4, box_color, -1, cv::LINE_AA);

            // 绘制 Track ID 文本
            std::string label = box.class_name;
            cv::putText(test_img, label, cv::Point(box.left, box.top - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv::LINE_AA);
        }

        // 6. 保存结果
        cv::imwrite("test_filter_boxes_in_polygon.jpg", test_img);
        std::cout << "Saved test_filter_boxes_in_polygon.jpg" << std::endl;

        // 7. 显示结果
        cv::imshow("YOLO Polygon Detection", test_img);
        cv::waitKey(0);
    }
}
