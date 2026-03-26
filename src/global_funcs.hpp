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

    GereralConfig read_config(const std::string &config_path)
    {
        GereralConfig config = {};

        nlohmann::json j;
        std::ifstream ifs(config_path);
        ifs >> j;
        ifs.close();

        if (j.contains("model_path"))
            config.detect_config.model_path = j["model_path"].get<std::string>();
        if (j.contains("imgsz"))
        {
            // [height, width]
            std::vector<int> imgsz = j["imgsz"].get<std::vector<int>>();
            if (imgsz.size() == 2)
            {
                config.detect_config.model_input_shape.width = imgsz[1];
                config.detect_config.model_input_shape.height = imgsz[0];
            }
        }
        if (j.contains("conf_threshold"))
            config.detect_config.conf_threshold = j["conf_threshold"].get<float>();
        if (j.contains("nms_threshold"))
            config.detect_config.nms_threshold = j["nms_threshold"].get<float>();
        if (j.contains("names"))
        {
            config.detect_config.classes.clear();
            for (auto &item : j["names"].items())
            {
                int id = std::stoi(item.key());
                std::string name = item.value().get<std::string>();
                config.detect_config.classes[id] = name;
            }
        }

        if (j.contains("track"))
        {
            auto t = j["track"];
            if (t.contains("max_time_lost"))
                config.track_config.max_time_lost = t["max_time_lost"].get<int>();
            if (t.contains("track_high_thresh"))
                config.track_config.track_high_thresh = t["track_high_thresh"].get<float>();
            if (t.contains("track_low_thresh"))
                config.track_config.track_low_thresh = t["track_low_thresh"].get<float>();
            if (t.contains("new_track_thresh"))
                config.track_config.new_track_thresh = t["new_track_thresh"].get<float>();
            if (t.contains("match_thresh"))
                config.track_config.match_thresh = t["match_thresh"].get<float>();
        }

        return config;
    }

}

#endif // GLOBAL_FUNCS_HPP
