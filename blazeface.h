//
// Created by aichao on 2022/4/26.
//

#ifndef BLAZEFACE_H
#define BLAZEFACE_H

#include <opencv2/opencv.hpp>
#include <net.h>
#include "cpu.h"
#include <benchmark.h>

struct FaceObject {
    cv::Rect_<float> rect;
    float prob;
};

class BlazeFace {
public:
    int load(int _target_size, bool use_gpu = false);

    int detect(const cv::Mat &rgb, std::vector<FaceObject> &faceobjects, float prob_threshold = 0.8f,
               float nms_threshold = 0.3f);

    int draw(cv::Mat &rgb, const std::vector<FaceObject> &faceobjects);


private:
    int generate_anchors(int target_size, int step_size, std::vector<float> min_sizes, std::vector<float> aspect_ratios,
                         float offset, std::vector<float> variances);

    void generate_proposals(const ncnn::Mat &score_blob, const ncnn::Mat &bbox_blob, float score_threshold,
                            int num_anchors, int target_size, std::vector<FaceObject> &faceobjects);

    const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
    const float norm_vals[3] = {0.017125f, 0.017507f, 0.017429f};
    const std::vector<std::vector<float>> min_sizes = {
            std::vector<float>{16.0, 24.0},
            std::vector<float>{32.0, 48.0, 64.0, 80.0, 96.0, 128.0}};
    const std::vector<std::vector<float>> aspect_ratios{
            std::vector<float>{1.0},
            std::vector<float>{1.0}};
    const std::vector<int> steps = {8, 16};
    const float offset = 0.5f;
    const std::vector<float> variances = {0.1, 0.1, 0.2, 0.2};
    std::vector<std::vector<float>> anchors = {};
    int target_size;
private:
    ncnn::Net blazeface;
    bool has_kps;
};

#endif // BLAZEFACE_H
