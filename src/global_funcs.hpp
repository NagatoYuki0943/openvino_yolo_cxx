#ifndef GLOBAL_FUNCS_HPP
#define GLOBAL_FUNCS_HPP
#pragma once

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "global_vars.hpp"

namespace Global
{

    GereralConfig read_config(const std::string &config_path);

    // 辅助函数：根据固定的 ID 生成稳定且明亮的颜色
    cv::Scalar GetColorForId(const int id);

    void draw_detected_object(cv::Mat &image, const std::vector<YoloDetectBox> &detect_boxes);

    std::map<int, std::vector<int>> classify_boxed_by_class(const std::vector<YoloDetectBox> &detect_boxes);
}

#endif // GLOBAL_FUNCS_HPP
