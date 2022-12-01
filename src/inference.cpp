//
// Created by ty on 2022/11/29.
//

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/core/utils/filesystem.hpp>
#include <ostream>
#include <vector>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>

#include <auto_log/autolog.h>
#include <gflags/gflags.h>
#include <include/cls.h>
#include <include/cls_config.h>
#include <include/inference.h>

using namespace std;
using namespace cv;
using namespace PaddleClas;

DEFINE_string(config,
              "", "Path of yaml file");
DEFINE_string(c,
              "", "Path of yaml file");

namespace Inference{
    ClsConfig clsInference::Init(int argc, char **argv){
        //      解析参数
        google::ParseCommandLineFlags(&argc, &argv, true);
        std::string yaml_path = "";
        std::cout << "run correctly" << std::endl;
        if (FLAGS_config == "" && FLAGS_c == "") {
            std::cerr << "[ERROR] usage: " << std::endl
                      << argv[0] << " -c $yaml_path" << std::endl
                      << "or:" << std::endl
                      << argv[0] << " -config $yaml_path" << std::endl;
            exit(1);
        } else if (FLAGS_config != "") {
            yaml_path = FLAGS_config;
        } else {
            yaml_path = FLAGS_c;
        }
        std::cout << "starting read configfile" << std::endl;
        ClsConfig config(yaml_path);
        std::cout << "read config complete" << std::endl;
//      config读取完毕
        return config;
    }
    void clsInference::Run(ClsConfig &config,const string &img_path){
        config.PrintConfigInfo();

        if (img_path.empty()) {

        }
        std::string path(config.infer_imgs);
        std::cout << "starting decode" << std::endl;
        std::vector <std::string> img_files_list;
        if (cv::utils::fs::isDirectory(path)) {
            std::vector <cv::String> filenames;
            cv::glob(path, filenames);
            for (auto f : filenames) {
                img_files_list.push_back(f);
            }
        } else {
            img_files_list.push_back(path);
        }

        std::cout << "img_file_list length: " << img_files_list.size() << std::endl;

        Classifier classifier(config);

        std::vector<double> cls_times = {0, 0, 0};
        std::vector<double> cls_times_total = {0, 0, 0};
        double infer_time;
        std::vector<float> out_data;
        std::vector<int> result_index;
        int warmup_iter = 5;
        bool label_output_equal_flag = true;
        for (int idx = 0; idx < img_files_list.size(); ++idx) {
            std::string img_path = img_files_list[idx];
            cv::Mat srcimg = cv::imread(img_path, cv::IMREAD_COLOR);
            if (!srcimg.data) {
                std::cerr << "[ERROR] image read failed! image path: " << img_path
                          << "\n";
                exit(-1);
            }

            cv::cvtColor(srcimg, srcimg, cv::COLOR_BGR2RGB);

            classifier.Run(srcimg, out_data, result_index, cls_times);

            if (label_output_equal_flag && out_data.size() != config.id_map.size()) {
                std::cout << "Warning: the label size is not equal to output size!"
                          << std::endl;
                label_output_equal_flag = false;
            }

            int max_len = std::min(config.topk, int(out_data.size()));
            std::cout << "Current image path: " << img_path << std::endl;
            infer_time = cls_times[0] + cls_times[1] + cls_times[2];
            std::cout << "Current total inferen time cost: " << infer_time << " ms."
                      << std::endl;
            for (int i = 0; i < max_len; ++i) {
                printf("\tTop%d: class_id: %d, score: %.4f, ", i + 1, result_index[i],
                       out_data[result_index[i]]);
                if (label_output_equal_flag)
                    printf("label: %s\n", config.id_map[result_index[i]].c_str());
            }
            std::cout << "\r\n======================end========================\r\n"<< std::endl;
        }
        if (img_files_list.size() > warmup_iter) {

            infer_time = cls_times_total[0] + cls_times_total[1] + cls_times_total[2];
            std::cout << "average time cost in all: "
                      << infer_time / (img_files_list.size() - warmup_iter) << " ms."
                      << std::endl;
        }

        std::string presion = "fp32";
        if (config.use_fp16)
            presion = "fp16";
        if (config.benchmark) {
            AutoLogger autolog("Classification", config.use_gpu, config.use_tensorrt,
                               config.use_mkldnn, config.cpu_threads, 1,
                               "1, 3, 224, 224", presion, cls_times_total,
                               img_files_list.size());
            autolog.report();
        }
//        return 0;
    }
}

