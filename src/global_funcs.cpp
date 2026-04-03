#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "global_vars.hpp"
#include "global_funcs.hpp"

namespace Global
{

    GereralConfig read_config(const std::string &config_path)
    {
        std::cout << "try read config from: " << config_path << std::endl;

        GereralConfig config = {};

        try
        {
            nlohmann::json j;
            std::ifstream ifs(config_path);
            ifs >> j;
            ifs.close();

            if (j.contains("model_path"))
            {
                config.detect_config.model_path = j["model_path"].get<std::string>();
                std::cout << "model_path: " << config.detect_config.model_path << std::endl;
            }
            if (j.contains("imgsz"))
            {
                // [height, width]
                std::vector<int> imgsz = j["imgsz"].get<std::vector<int>>();
                if (imgsz.size() == 2)
                {
                    config.detect_config.model_input_shape.width = imgsz[1];
                    config.detect_config.model_input_shape.height = imgsz[0];
                }
                std::cout << "imgsz height: " << config.detect_config.model_input_shape.height << ", width: " << config.detect_config.model_input_shape.width << std::endl;
            }
            if (j.contains("conf_threshold"))
            {
                config.detect_config.conf_threshold = j["conf_threshold"].get<float>();
                std::cout << "conf_threshold: " << config.detect_config.conf_threshold << std::endl;
            }
            if (j.contains("nms_threshold"))
            {
                config.detect_config.nms_threshold = j["nms_threshold"].get<float>();
                std::cout << "nms_threshold: " << config.detect_config.nms_threshold << std::endl;
            }
            if (j.contains("names"))
            {
                config.detect_config.classes.clear();
                std::cout << "id2names: " << std::endl;
                for (auto &item : j["names"].items())
                {
                    int id = std::stoi(item.key());
                    std::string name = item.value().get<std::string>();
                    config.detect_config.classes[id] = name;
                    std::cout << "    id: " << id << ", name: " << name << std::endl;
                }
            }

            if (j.contains("track"))
            {
                auto t = j["track"];
                if (t.contains("max_time_lost"))
                {
                    config.track_config.max_time_lost = t["max_time_lost"].get<int>();
                    std::cout << "track max_time_lost: " << config.track_config.max_time_lost << std::endl;
                }
                if (t.contains("track_high_thresh"))
                {
                    config.track_config.track_high_thresh = t["track_high_thresh"].get<float>();
                    std::cout << "track track_high_thresh: " << config.track_config.track_high_thresh << std::endl;
                }
                if (t.contains("track_low_thresh"))
                {
                    config.track_config.track_low_thresh = t["track_low_thresh"].get<float>();
                    std::cout << "track track_low_thresh: " << config.track_config.track_low_thresh << std::endl;
                }
                if (t.contains("new_track_thresh"))
                {
                    config.track_config.new_track_thresh = t["new_track_thresh"].get<float>();
                    std::cout << "track new_track_thresh: " << config.track_config.new_track_thresh << std::endl;
                }
                if (t.contains("match_thresh"))
                {
                    config.track_config.match_thresh = t["match_thresh"].get<float>();
                    std::cout << "track match_thresh: " << config.track_config.match_thresh << std::endl;
                }
                if (t.contains("min_hits"))
                {
                    config.track_config.min_hits = t["min_hits"].get<int>();
                    std::cout << "track min_hits: " << config.track_config.min_hits << std::endl;
                }
            }

            std::cout << "read config from: " << config_path << " success" << std::endl;

            return config;
        }
        catch (const std::exception &e)
        {
            std::cout << "read config from: " << config_path << " fail" << std::endl;
            std::cerr << e.what() << '\n';
            return config;
        }
    }

}
