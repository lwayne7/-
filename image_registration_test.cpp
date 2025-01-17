#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#include <numeric>

// 图像配准类
class ImageRegistration {
public:
    // 特征检测器类型
    enum DetectorType {
        DETECTOR_ORB,
        DETECTOR_SIFT,
        DETECTOR_SURF
    };

    ImageRegistration() {}
    
    // 创建特征检测器
    cv::Ptr<cv::Feature2D> createDetector(DetectorType type) {
        switch (type) {
            case DETECTOR_ORB:
                return cv::ORB::create(
                    3000,               // nfeatures
                    1.2f,              // scaleFactor
                    8,                 // nlevels
                    31,                // edgeThreshold
                    0,                 // firstLevel
                    2,                 // WTA_K
                    cv::ORB::HARRIS_SCORE,  // scoreType
                    31,                // patchSize
                    20                 // fastThreshold
                );
            case DETECTOR_SIFT:
                return cv::SIFT::create(
                    3000,              // nfeatures
                    3,                 // nOctaveLayers
                    0.04,              // contrastThreshold
                    10,               // edgeThreshold
                    1.6               // sigma
                );
            case DETECTOR_SURF:
                return cv::xfeatures2d::SURF::create(
                    100,               // hessianThreshold
                    4,                 // nOctaves
                    3,                 // nOctaveLayers
                    false,             // extended
                    false              // upright
                );
            default:
                throw std::runtime_error("未知的特征检测器类型");
        }
    }
    
    // 使用特征点匹配法进行图像配准
    cv::Mat registerByFeature(const cv::Mat& img, const cv::Mat& ref,
                            double& angle, double& scale,
                            cv::Point2d& translation,
                            DetectorType detector_type,
                            std::string matches_output_path) {
        // 预处理图像
        cv::Mat img1 = preprocessImage(img);
        cv::Mat img2 = preprocessImage(ref);
        
        std::cout << "预处理后图像类型: " << img1.type() << std::endl;
        
        try {
            // 创建特征检测器
            cv::Ptr<cv::Feature2D> detector = createDetector(detector_type);
            
            // 检测关键点和计算描述子
            std::vector<cv::KeyPoint> keypoints1, keypoints2;
            cv::Mat descriptors1, descriptors2;
            
            detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
            std::cout << "第一张图像检测到 " << keypoints1.size() << " 个关键点" << std::endl;
            
            detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
            std::cout << "第二张图像检测到 " << keypoints2.size() << " 个关键点" << std::endl;
            
            if (keypoints1.empty() || keypoints2.empty()) {
                throw std::runtime_error("未检测到关键点");
            }
            
            // 根据检测器类型选择合适的匹配器
            cv::Ptr<cv::DescriptorMatcher> matcher;
            if (detector_type == DETECTOR_ORB) {
                matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
            } else {
                matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
            }
            
            // 使用KNN匹配
            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
            
            std::cout << "找到 " << knn_matches.size() << " 组KNN匹配" << std::endl;
            
            // 使用比率测试进行初步过滤
            const float ratio_thresh = 0.75f;
            std::vector<cv::DMatch> good_matches;
            for (size_t i = 0; i < knn_matches.size(); i++) {
                if (knn_matches[i].size() >= 2) {
                    if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                        good_matches.push_back(knn_matches[i][0]);
                    }
                }
            }
            
            std::cout << "比率测试后剩余 " << good_matches.size() << " 个匹配" << std::endl;
            
            // 根据匹配点距离分布进行进一步过滤
            if (!good_matches.empty()) {
                std::vector<float> distances;
                for (const auto& match : good_matches) {
                    distances.push_back(match.distance);
                }
                
                // 计算距离的中位数和标准差
                std::sort(distances.begin(), distances.end());
                float median = distances[distances.size()/2];
                float sum = std::accumulate(distances.begin(), distances.end(), 0.0f);
                float mean = sum / distances.size();
                float sq_sum = std::inner_product(distances.begin(), distances.end(), distances.begin(), 0.0f);
                float stdev = std::sqrt(sq_sum / distances.size() - mean * mean);
                
                // 使用3倍标准差规则过滤异常值
                std::vector<cv::DMatch> better_matches;
                float upper_bound = median + 3 * stdev;
                for (const auto& match : good_matches) {
                    if (match.distance <= upper_bound) {
                        better_matches.push_back(match);
                    }
                }
                good_matches = better_matches;
            }
            
            std::cout << "距离过滤后剩余 " << good_matches.size() << " 个匹配" << std::endl;
            
            if (good_matches.size() < 4) {
                throw std::runtime_error("没有足够的好匹配点来计算变换");
            }
            
            // 绘制匹配结果
            if (!matches_output_path.empty()) {
                cv::Mat matches_visualization;
                cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, matches_visualization,
                               cv::Scalar::all(-1), cv::Scalar::all(-1),
                               std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                cv::imwrite(matches_output_path, matches_visualization);
            }
            
            // 获取匹配点的坐标
            std::vector<cv::Point2f> points1, points2;
            for (size_t i = 0; i < good_matches.size(); i++) {
                points1.push_back(keypoints1[good_matches[i].queryIdx].pt);
                points2.push_back(keypoints2[good_matches[i].trainIdx].pt);
            }
            
            // 使用RANSAC方法计算仿射变换矩阵,并获取内点掩码
            cv::Mat inliers_mask;
            cv::Mat affineMatrix = cv::estimateAffinePartial2D(
                points2, points1,  // 从参考图像到失真图像的变换
                inliers_mask,      // inliers mask
                cv::RANSAC,        // method
                3.0,               // ransacReprojThreshold
                2000,              // maxIters
                0.99,              // confidence
                20                 // refineIters
            );
            
            // 使用内点掩码过滤匹配点
            std::vector<cv::DMatch> inlier_matches;
            for (size_t i = 0; i < good_matches.size(); i++) {
                if (inliers_mask.at<uchar>(i)) {
                    inlier_matches.push_back(good_matches[i]);
                }
            }
            
            std::cout << "RANSAC内点数量: " << inlier_matches.size() << std::endl;
            
            // 使用内点重新计算变换矩阵
            points1.clear();
            points2.clear();
            for (size_t i = 0; i < inlier_matches.size(); i++) {
                points1.push_back(keypoints1[inlier_matches[i].queryIdx].pt);
                points2.push_back(keypoints2[inlier_matches[i].trainIdx].pt);
            }
            
            // 重新计算变换矩阵（从参考图像到失真图像的变换）
            affineMatrix = cv::estimateAffinePartial2D(points2, points1);
            
            if (affineMatrix.empty()) {
                throw std::runtime_error("无法计算仿射变换矩阵");
            }
            
            // 从仿射变换矩阵中提取变换参数
            double a = affineMatrix.at<double>(0,0);
            double b = affineMatrix.at<double>(0,1);
            
            // 计算缩放和旋转
            scale = sqrt(a*a + b*b);  // 缩放比例
            angle = atan2(b, a) * 180 / CV_PI;  // 旋转角度
            translation = cv::Point2d(affineMatrix.at<double>(0,2), affineMatrix.at<double>(1,2));
            
            // 计算逆变换矩阵
            cv::Mat inverseMatrix;
            cv::invertAffineTransform(affineMatrix, inverseMatrix);
            
            // 计算输出图像的尺寸
            cv::Point2f corners[4] = {
                cv::Point2f(0, 0),
                cv::Point2f(img.cols - 1, 0),
                cv::Point2f(img.cols - 1, img.rows - 1),
                cv::Point2f(0, img.rows - 1)
            };
            
            // 使用逆变换矩阵变换角点
            std::vector<cv::Point2f> transformed_corners(4);
            cv::transform(std::vector<cv::Point2f>(corners, corners + 4), transformed_corners, inverseMatrix);
            
            // 计算变换后图像的边界
            float min_x = std::numeric_limits<float>::max();
            float min_y = std::numeric_limits<float>::max();
            float max_x = std::numeric_limits<float>::lowest();
            float max_y = std::numeric_limits<float>::lowest();
            
            for (const auto& point : transformed_corners) {
                min_x = std::min(min_x, point.x);
                min_y = std::min(min_y, point.y);
                max_x = std::max(max_x, point.x);
                max_y = std::max(max_y, point.y);
            }
            
            // 计算新的尺寸和平移量
            int new_width = static_cast<int>(std::ceil(max_x - min_x));
            int new_height = static_cast<int>(std::ceil(max_y - min_y));
            
            // 确保尺寸是偶数
            new_width += new_width % 2;
            new_height += new_height % 2;
            
            // 调整变换矩阵以考虑边界平移
            cv::Mat adjustment = cv::Mat::eye(3, 3, CV_64F);
            adjustment.at<double>(0,2) = -min_x;
            adjustment.at<double>(1,2) = -min_y;
            
            // 将仿射矩阵转换为3x3形式
            cv::Mat affine3x3 = cv::Mat::eye(3, 3, CV_64F);
            inverseMatrix.copyTo(affine3x3(cv::Rect(0, 0, 3, 2)));
            
            // 组合变换
            cv::Mat finalTransform = adjustment * affine3x3;
            
            // 提取最终的2x3仿射矩阵
            cv::Mat finalAffine = finalTransform(cv::Rect(0, 0, 3, 2));
            
            // 应用变换
            cv::Mat registered;
            cv::warpAffine(img, registered, finalAffine, cv::Size(new_width, new_height), 
                          cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
            
            // 保存中间结果
            cv::Mat comparison;
            
            // 将所有图像调整为相同的尺寸
            cv::Mat resized_distorted, resized_registered, resized_ref;
            int max_height = std::max({img.rows, registered.rows, ref.rows});
            
            // 保持宽高比调整图像大小
            double dist_scale = static_cast<double>(max_height) / img.rows;
            cv::resize(img, resized_distorted, cv::Size(), dist_scale, dist_scale, cv::INTER_LINEAR);
            
            double reg_scale = static_cast<double>(max_height) / registered.rows;
            cv::resize(registered, resized_registered, cv::Size(), reg_scale, reg_scale, cv::INTER_LINEAR);
            
            double ref_scale = static_cast<double>(max_height) / ref.rows;
            cv::resize(ref, resized_ref, cv::Size(), ref_scale, ref_scale, cv::INTER_LINEAR);
            
            // 计算最大宽度
            int max_width = std::max({resized_distorted.cols, resized_registered.cols, resized_ref.cols});
            
            // 创建画布
            cv::Mat canvas = cv::Mat::zeros(max_height, max_width * 3, resized_distorted.type());
            
            // 在画布上绘制图像
            resized_distorted.copyTo(canvas(cv::Rect(0, 0, resized_distorted.cols, resized_distorted.rows)));
            resized_registered.copyTo(canvas(cv::Rect(max_width, 0, resized_registered.cols, resized_registered.rows)));
            resized_ref.copyTo(canvas(cv::Rect(max_width * 2, 0, resized_ref.cols, resized_ref.rows)));
            
            cv::imwrite("comparison.jpg", canvas);
            
            return registered;
        } catch (const cv::Exception& e) {
            std::cout << "OpenCV异常: " << e.what() << std::endl;
            throw;
        } catch (const std::exception& e) {
            std::cout << "标准异常: " << e.what() << std::endl;
            throw;
        }
    }
    
private:
    // 图像预处理
    cv::Mat preprocessImage(const cv::Mat& img) {
        cv::Mat result;
        
        // 转换为灰度图像
        if(img.channels() > 1) {
            cv::cvtColor(img, result, cv::COLOR_BGR2GRAY);
        } else {
            result = img.clone();
        }
        
        // 确保图像类型为CV_8U
        if(result.type() != CV_8U) {
            result.convertTo(result, CV_8U);
        }
        
        return result;
    }
};

// 主函数
int main() {
    // 创建图像配准系统实例
    ImageRegistration registrator;
    
    // 读取原始图像
    cv::Mat host = cv::imread("syn_copy_256.png", cv::IMREAD_COLOR);
    //cv::Mat host = cv::imread("lena.jpg", cv::IMREAD_COLOR);
    if(host.empty()) {
        std::cout << "无法读取原始图像: syn_copy_256.png" << std::endl;
        return -1;
    }
    
    // 转换为灰度图
    cv::Mat host_gray;
    cv::cvtColor(host, host_gray, cv::COLOR_BGR2GRAY);
    
    // 定义要测试的旋转角度和缩放比例组合
    struct TestCase {
        double angle;
        double scale;
        std::string description;
    };
    
    std::vector<TestCase> test_cases = {
        {15, 0.8, "小角度旋转+轻微缩小"},
        {45, 1.2, "中等角度旋转+轻微放大"},
        {90, 1.5, "大角度旋转+中等放大"},
        {180, 0.5, "极限角度旋转+过度缩小"}
    };
    
    // 定义要测试的特征检测器
    std::vector<std::pair<ImageRegistration::DetectorType, std::string>> detectors = {
        {ImageRegistration::DETECTOR_ORB, "ORB"},
        {ImageRegistration::DETECTOR_SIFT, "SIFT"},
        {ImageRegistration::DETECTOR_SURF, "SURF"}
    };
    
    // 定义两种测试方法
    enum TestMethod {
        METHOD_FIXED_SIZE,    // 固定尺寸方法（原始方法）
        METHOD_DYNAMIC_SIZE   // 动态尺寸方法（防止信息丢失）
    };
    
    std::vector<std::pair<TestMethod, std::string>> methods = {
        {METHOD_FIXED_SIZE, "fixed"},
        {METHOD_DYNAMIC_SIZE, "dynamic"}
    };
    
    // 对每种方法进行测试
    for (const auto& method : methods) {
        std::cout << "\n\n=== 测试方法: " << (method.first == METHOD_FIXED_SIZE ? "固定尺寸" : "动态尺寸") << " ===" << std::endl;
        
        // 对每种特征检测器进行测试
        for (const auto& detector : detectors) {
            std::cout << "\n=== " << detector.second << "特征检测器测试结果 ===" << std::endl;
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "测试场景\t预期角度\t预期缩放\t检测角度\t检测缩放\t角度误差\t缩放误差\tPSNR(dB)" << std::endl;
            
            // 测试每种组合
            for(const auto& test : test_cases) {
                std::cout << "\n测试场景: " << test.description << std::endl;
                
                // 对原始图像进行旋转和缩放
                cv::Point2f center(host_gray.cols/2.0f, host_gray.rows/2.0f);
                cv::Mat rotationMatrix;
                cv::Mat distorted;
                
                if (method.first == METHOD_FIXED_SIZE) {
                    // 固定尺寸方法
                    rotationMatrix = cv::getRotationMatrix2D(center, test.angle, test.scale);
                    cv::warpAffine(host_gray, distorted, rotationMatrix, host_gray.size());
                } else {
                    // 动态尺寸方法
                    double angle_rad = test.angle * CV_PI / 180.0;
                    double abs_cos = std::abs(std::cos(angle_rad));
                    double abs_sin = std::abs(std::sin(angle_rad));
                    
                    int new_width = static_cast<int>((host_gray.cols * abs_cos + host_gray.rows * abs_sin) * test.scale);
                    int new_height = static_cast<int>((host_gray.cols * abs_sin + host_gray.rows * abs_cos) * test.scale);
                    
                    new_width += new_width % 2;
                    new_height += new_height % 2;
                    
                    rotationMatrix = cv::getRotationMatrix2D(center, test.angle, test.scale);
                    rotationMatrix.at<double>(0,2) += (new_width - host_gray.cols) / 2.0;
                    rotationMatrix.at<double>(1,2) += (new_height - host_gray.rows) / 2.0;
                    
                    cv::warpAffine(host_gray, distorted, rotationMatrix, cv::Size(new_width, new_height));
                }
                
                // 保存失真图像
                std::string distorted_filename = detector.second + "_" + method.second + "_distorted_a" + 
                                               std::to_string(int(test.angle)) + 
                                               "_s" + std::to_string(int(test.scale*100)) + ".jpg";
                cv::imwrite(distorted_filename, distorted);
                
                // 使用特征点匹配法进行配准
                double detectedAngle, detectedScale;
                cv::Point2d detectedTranslation;
                std::string matches_vis_path;
                
                try {
                    // 构建特征匹配可视化输出路径
                    matches_vis_path = "feature_matches_" + detector.second + "_" + method.second + "_a" + 
                                       std::to_string(int(test.angle)) + 
                                       "_s" + std::to_string(int(test.scale*100)) + ".jpg";
                    
                    cv::Mat registeredFeature = registrator.registerByFeature(
                        distorted, host_gray,
                        detectedAngle, detectedScale,
                        detectedTranslation,
                        detector.first,
                        matches_vis_path
                    );
                    
                    // 计算误差
                    double angleError = std::abs(detectedAngle - test.angle);
                    double scaleError = std::abs(detectedScale - test.scale) / test.scale * 100.0;
                    
                    // 计算PSNR
                    double psnr;
                    if (registeredFeature.size() != host_gray.size()) {
                        cv::Mat resized_registered;
                        cv::resize(registeredFeature, resized_registered, host_gray.size(), 0, 0, cv::INTER_LINEAR);
                        psnr = cv::PSNR(resized_registered, host_gray);
                    } else {
                        psnr = cv::PSNR(registeredFeature, host_gray);
                    }
                    
                    // 输出结果
                    std::cout << test.description << "\t"
                             << test.angle << "°\t" 
                             << test.scale << "\t" 
                             << detectedAngle << "°\t"
                             << detectedScale << "\t"
                             << angleError << "°\t"
                             << scaleError << "%\t"
                             << psnr << std::endl;
                    
                    // 保存配准结果的对比图
                    cv::Mat comparison;
                    cv::Mat resized_distorted, resized_registered, resized_ref;
                    int max_height = std::max({distorted.rows, registeredFeature.rows, host_gray.rows});
                    
                    // 保持宽高比调整图像大小
                    double dist_scale = static_cast<double>(max_height) / distorted.rows;
                    cv::resize(distorted, resized_distorted, cv::Size(), dist_scale, dist_scale, cv::INTER_LINEAR);
                    
                    double reg_scale = static_cast<double>(max_height) / registeredFeature.rows;
                    cv::resize(registeredFeature, resized_registered, cv::Size(), reg_scale, reg_scale, cv::INTER_LINEAR);
                    
                    double ref_scale = static_cast<double>(max_height) / host_gray.rows;
                    cv::resize(host_gray, resized_ref, cv::Size(), ref_scale, ref_scale, cv::INTER_LINEAR);
                    
                    // 计算最大宽度和总宽度
                    int max_width = std::max({resized_distorted.cols, resized_registered.cols, resized_ref.cols});
                    int total_width = max_width * 3 + 20;  // 添加一些间距
                    
                    // 创建画布
                    cv::Mat canvas = cv::Mat::zeros(max_height, total_width, resized_distorted.type());
                    
                    // 计算每个图像的起始x坐标，使其居中
                    int x1 = (max_width - resized_distorted.cols) / 2;
                    int x2 = max_width + 10 + (max_width - resized_registered.cols) / 2;
                    int x3 = 2 * max_width + 20 + (max_width - resized_ref.cols) / 2;
                    
                    // 在画布上绘制图像
                    resized_distorted.copyTo(canvas(cv::Rect(x1, 0, resized_distorted.cols, resized_distorted.rows)));
                    resized_registered.copyTo(canvas(cv::Rect(x2, 0, resized_registered.cols, resized_registered.rows)));
                    resized_ref.copyTo(canvas(cv::Rect(x3, 0, resized_ref.cols, resized_ref.rows)));
                    
                    cv::imwrite("registration_result_" + detector.second + "_" + method.second + "_a" + 
                                std::to_string(int(test.angle)) + 
                                "_s" + std::to_string(int(test.scale*100)) + ".jpg", canvas);
                    
                    // 单独保存配准后的图像
                    cv::imwrite("registered_" + detector.second + "_" + method.second + "_a" + 
                                std::to_string(int(test.angle)) + 
                                "_s" + std::to_string(int(test.scale*100)) + ".jpg", registeredFeature);
                    
                } catch (const std::exception& e) {
                    std::cout << "配准失败: " << e.what() << std::endl;
                }
            }
        }
    }
    
    return 0;
} 