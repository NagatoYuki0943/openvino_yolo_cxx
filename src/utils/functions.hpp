#ifndef DETECT_UTILS_HPP
#define DETECT_UTILS_HPP
#pragma once

#include <string>
#include <vector>
#include <map>
#include "../global_vars.hpp"


namespace detect_utils
{

    // 辅助函数：根据固定的 ID 生成稳定且明亮的颜色
    cv::Scalar GetColorForId(const int id);

    void draw_detected_object(cv::Mat &image, const std::vector<Global::YoloDetectBox> &detect_boxes);

    std::map<int, std::vector<int>> classify_boxed_by_class(const std::vector<Global::YoloDetectBox> &detect_boxes);

}

#endif // DETECT_UTILS_HPP
