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

}

#endif // GLOBAL_FUNCS_HPP
