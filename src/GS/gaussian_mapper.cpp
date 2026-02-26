/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "include/gaussian_mapper.h"

GaussianMapper::GaussianMapper(
    std::shared_ptr<dso::FullSystem> pSLAM,
    std::filesystem::path gaussian_config_file_path,
    std::filesystem::path result_dir,
    int seed,
    torch::DeviceType device_type)
    : pSLAM_(pSLAM),
      initial_mapped_(false),
      interrupt_training_(false),
      stopped_(false),
      iteration_(0),
      ema_loss_for_log_(0.0f),
      SLAM_ended_(false),
      loop_closure_iteration_(false),
      min_num_initial_map_kfs_(15UL),
      large_rot_th_(1e-1f),
      large_trans_th_(1e-2f),
      training_report_interval_(0)
{
    // Random seed
    std::srand(seed);
    torch::manual_seed(seed);

    // Device
    if (device_type == torch::kCUDA && torch::cuda::is_available()) {
        std::cout << "[Gaussian Mapper]CUDA available! Training on GPU." << std::endl;
        device_type_ = torch::kCUDA;
        model_params_.data_device_ = "cuda";
    }
    else {
        std::cout << "[Gaussian Mapper]Training on CPU." << std::endl;
        device_type_ = torch::kCPU;
        model_params_.data_device_ = "cpu";
    }

    result_dir_ = result_dir;
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    config_file_path_ = gaussian_config_file_path;
    readConfigFromFile(gaussian_config_file_path);

    std::vector<float> bg_color;
    if (model_params_.white_background_)
        bg_color = {1.0f, 1.0f, 1.0f};
    else
        bg_color = {0.0f, 0.0f, 0.0f};
    background_ = torch::tensor(bg_color,
                    torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

    // Initialize scene and model
    gaussians_ = std::make_shared<GaussianModel>(model_params_);
    scene_ = std::make_shared<GaussianScene>(model_params_);

    // Mode
    if (!pSLAM) {
        // NO SLAM
        return;
    }

    // Cameras
    // TODO: not only monocular (from photo-slam)
    cv::Size SLAM_im_size(image_width, image_height);

    std::vector<float> vPinHoleDistorsion1(5);
    vPinHoleDistorsion1[0] = distortion_k1; // k1
    vPinHoleDistorsion1[1] = distortion_k2; // k2
    vPinHoleDistorsion1[2] = distortion_p1; // p1
    vPinHoleDistorsion1[3] = distortion_p2; // p2
    vPinHoleDistorsion1[4] = distortion_k3; // k3
    cv::Mat camera1DistortionCoef = cv::Mat(vPinHoleDistorsion1.size(),1,CV_32F,vPinHoleDistorsion1.data());
    UndistortParams undistort_params(
        SLAM_im_size,
        // settings->camera1DistortionCoef()
        camera1DistortionCoef
    );
    
    if (distortion_k1 != 0 || distortion_k2 != 0 || distortion_p1 != 0 || distortion_p2 != 0 || distortion_k3 != 0) {
        need_distortion = true;
    }

    Camera camera;
    camera.camera_id_ = 0;
    camera.setModelId(Camera::CameraModelType::PINHOLE);
    
    // float SLAM_fx = pSLAM_->Hcalib.fxl();
    // float SLAM_fy = pSLAM_->Hcalib.fyl();
    // float SLAM_cx = pSLAM_->Hcalib.cxl();
    // float SLAM_cy = pSLAM_->Hcalib.cyl();
    float SLAM_fx = mapping_fx;
    float SLAM_fy = mapping_fy;
    float SLAM_cx = mapping_cx;
    float SLAM_cy = mapping_cy;

    // Old K, i.e. K in SLAM
    cv::Mat K = (
        cv::Mat_<float>(3, 3)
            << SLAM_fx, 0.f, SLAM_cx,
                0.f, SLAM_fy, SLAM_cy,
                0.f, 0.f, 1.f
    );

    camera.width_ = undistort_params.old_size_.width;
    float x_ratio = static_cast<float>(camera.width_) / undistort_params.old_size_.width;

    camera.height_ = undistort_params.old_size_.height;
    float y_ratio = static_cast<float>(camera.height_) / undistort_params.old_size_.height;

    camera.num_gaus_pyramid_sub_levels_ = num_gaus_pyramid_sub_levels_;
    camera.gaus_pyramid_width_.resize(num_gaus_pyramid_sub_levels_);
    camera.gaus_pyramid_height_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        camera.gaus_pyramid_width_[l] = camera.width_ * this->kf_gaus_pyramid_factors_[l];
        camera.gaus_pyramid_height_[l] = camera.height_ * this->kf_gaus_pyramid_factors_[l];
    }

    camera.params_[0]/*new fx*/= SLAM_fx * x_ratio;
    camera.params_[1]/*new fy*/= SLAM_fy * y_ratio;
    camera.params_[2]/*new cx*/= SLAM_cx * x_ratio;
    camera.params_[3]/*new cy*/= SLAM_cy * y_ratio;

    cv::Mat K_new = (
        cv::Mat_<float>(3, 3)
            << camera.params_[0], 0.f, camera.params_[2],
                0.f, camera.params_[1], camera.params_[3],
                0.f, 0.f, 1.f
    );

    // Undistortion
    // if (this->sensor_type_ == MONOCULAR || this->sensor_type_ == RGBD)
    undistort_params.dist_coeff_.copyTo(camera.dist_coeff_);

    camera.initUndistortRectifyMapAndMask(K, SLAM_im_size, K_new, true);

    undistort_mask_[camera.camera_id_] =
        tensor_utils::cvMat2TorchTensor_Float32(
            camera.undistort_mask, device_type_);

    cv::Mat viewer_sub_undistort_mask;
    int viewer_image_height_ = camera.height_ * rendered_image_viewer_scale_;
    int viewer_image_width_ = camera.width_ * rendered_image_viewer_scale_;
    cv::resize(camera.undistort_mask, viewer_sub_undistort_mask,
                cv::Size(viewer_image_width_, viewer_image_height_));
    viewer_sub_undistort_mask_[camera.camera_id_] =
        tensor_utils::cvMat2TorchTensor_Float32(
            viewer_sub_undistort_mask, device_type_);

    cv::Mat viewer_main_undistort_mask;
    int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
    int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
    cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                cv::Size(viewer_image_width_main_, viewer_image_height_main_));
    viewer_main_undistort_mask_[camera.camera_id_] =
        tensor_utils::cvMat2TorchTensor_Float32(
            viewer_main_undistort_mask, device_type_);

    if (!viewer_camera_id_set_) {
        viewer_camera_id_ = camera.camera_id_;
        viewer_camera_id_set_ = true;
    }

    this->scene_->addCamera(camera);
}

void GaussianMapper::readConfigFromFile(std::filesystem::path cfg_path)
{
    cv::FileStorage settings_file(cfg_path.string().c_str(), cv::FileStorage::READ);
    if(!settings_file.isOpened()) {
       std::cerr << "[Gaussian Mapper]Failed to open settings file at: " << cfg_path << std::endl;
       exit(-1);
    }

    std::cout << "[Gaussian Mapper]Reading parameters from " << cfg_path << std::endl;
    std::unique_lock<std::mutex> lock(mutex_settings_);

    // SLAM parameters
    image_width = 
        settings_file["SLAM.image_width"].operator int();
    image_height = 
        settings_file["SLAM.image_height"].operator int();
    distortion_k1 =
        settings_file["SLAM.distortion_k1"].operator float();
    distortion_k2 =
        settings_file["SLAM.distortion_k2"].operator float();
    distortion_p1 =
        settings_file["SLAM.distortion_p1"].operator float();
    distortion_p2 =
        settings_file["SLAM.distortion_p2"].operator float();
    distortion_k3 =
        settings_file["SLAM.distortion_k3"].operator float();
    // depth_scale =
    //     settings_file["SLAM.depth_scale"].operator float();
    mapping_fx =
        settings_file["Mapper.fx"].operator float();
    mapping_fy =
        settings_file["Mapper.fy"].operator float();
    mapping_cx =
        settings_file["Mapper.cx"].operator float();
    mapping_cy =
        settings_file["Mapper.cy"].operator float();

    // added
    lambda_sparse_depth =
        settings_file["Mapper.lambda_sparse_depth"].operator float();

    // Model parameters
    model_params_.sh_degree_ =
        settings_file["Model.sh_degree"].operator int();
    model_params_.resolution_ =
        settings_file["Model.resolution"].operator float();
    model_params_.white_background_ =
        (settings_file["Model.white_background"].operator int()) != 0;
    model_params_.eval_ =
        (settings_file["Model.eval"].operator int()) != 0;

    // Pipeline Parameters
    z_near_ =
        settings_file["Camera.z_near"].operator float();
    z_far_ =
        settings_file["Camera.z_far"].operator float();

    monocular_inactive_geo_densify_max_pixel_dist_ =
        settings_file["Monocular.inactive_geo_densify_max_pixel_dist"].operator float();
    stereo_min_disparity_ =
        settings_file["Stereo.min_disparity"].operator int();
    stereo_num_disparity_ =
        settings_file["Stereo.num_disparity"].operator int();
    RGBD_min_depth_ =
        settings_file["RGBD.min_depth"].operator float();
    RGBD_max_depth_ =
        settings_file["RGBD.max_depth"].operator float();

    inactive_geo_densify_ =
        (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
    max_depth_cached_ =
        settings_file["Mapper.depth_cache"].operator int();
    min_num_initial_map_kfs_ = 
        static_cast<unsigned long>(settings_file["Mapper.min_num_initial_map_kfs"].operator int());
    new_keyframe_times_of_use_ = 
        settings_file["Mapper.new_keyframe_times_of_use"].operator int();
    local_BA_increased_times_of_use_ = 
        settings_file["Mapper.local_BA_increased_times_of_use"].operator int();
    loop_closure_increased_times_of_use_ = 
        settings_file["Mapper.loop_closure_increased_times_of_use_"].operator int();
    cull_keyframes_ =
        (settings_file["Mapper.cull_keyframes"].operator int()) != 0;
    large_rot_th_ =
        settings_file["Mapper.large_rotation_threshold"].operator float();
    large_trans_th_ =
        settings_file["Mapper.large_translation_threshold"].operator float();
    stable_num_iter_existence_ =
        settings_file["Mapper.stable_num_iter_existence"].operator int();

    pipe_params_.convert_SHs_ =
        (settings_file["Pipeline.convert_SHs"].operator int()) != 0;
    pipe_params_.compute_cov3D_ =
        (settings_file["Pipeline.compute_cov3D"].operator int()) != 0;

    do_gaus_pyramid_training_ =
        (settings_file["GausPyramid.do"].operator int()) != 0;
    num_gaus_pyramid_sub_levels_ =
        settings_file["GausPyramid.num_sub_levels"].operator int();
    int sub_level_times_of_use =
        settings_file["GausPyramid.sub_level_times_of_use"].operator int();
    kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
    kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
        kf_gaus_pyramid_factors_[l] = std::pow(0.5f, num_gaus_pyramid_sub_levels_ - l);
    }

    keyframe_record_interval_ = 
        settings_file["Record.keyframe_record_interval"].operator int();
    all_keyframes_record_interval_ = 
        settings_file["Record.all_keyframes_record_interval"].operator int();
    record_rendered_image_ = 
        (settings_file["Record.record_rendered_image"].operator int()) != 0;
    record_ground_truth_image_ = 
        (settings_file["Record.record_ground_truth_image"].operator int()) != 0;
    record_loss_image_ = 
        (settings_file["Record.record_loss_image"].operator int()) != 0;
    record_depth_image_ = 
        (settings_file["Record.record_depth_image"].operator int()) != 0;
    training_report_interval_ = 
        settings_file["Record.training_report_interval"].operator int();
    record_loop_ply_ =
        (settings_file["Record.record_loop_ply"].operator int()) != 0;

    // Optimization Parameters
    opt_params_.iterations_ =
        settings_file["Optimization.max_num_iterations"].operator int();
    opt_params_.position_lr_init_ =
        settings_file["Optimization.position_lr_init"].operator float();
    opt_params_.position_lr_final_ =
        settings_file["Optimization.position_lr_final"].operator float();
    opt_params_.position_lr_delay_mult_ =
        settings_file["Optimization.position_lr_delay_mult"].operator float();
    opt_params_.position_lr_max_steps_ =
        settings_file["Optimization.position_lr_max_steps"].operator int();
    opt_params_.feature_lr_ =
        settings_file["Optimization.feature_lr"].operator float();
    opt_params_.opacity_lr_ =
        settings_file["Optimization.opacity_lr"].operator float();
    opt_params_.scaling_lr_ =
        settings_file["Optimization.scaling_lr"].operator float();
    opt_params_.rotation_lr_ =
        settings_file["Optimization.rotation_lr"].operator float();

    opt_params_.percent_dense_ =
        settings_file["Optimization.percent_dense"].operator float();
    opt_params_.lambda_dssim_ =
        settings_file["Optimization.lambda_dssim"].operator float();
    opt_params_.densification_interval_ =
        settings_file["Optimization.densification_interval"].operator int();
    opt_params_.opacity_reset_interval_ =
        settings_file["Optimization.opacity_reset_interval"].operator int();
    opt_params_.densify_from_iter_ =
        settings_file["Optimization.densify_from_iter_"].operator int();
    opt_params_.densify_until_iter_ =
        settings_file["Optimization.densify_until_iter"].operator int();
    opt_params_.densify_grad_threshold_ =
        settings_file["Optimization.densify_grad_threshold"].operator float();

    prune_big_point_after_iter_ =
        settings_file["Optimization.prune_big_point_after_iter"].operator int();
    densify_min_opacity_ =
        settings_file["Optimization.densify_min_opacity"].operator float();

    // Viewer Parameters
    rendered_image_viewer_scale_ =
        settings_file["GaussianViewer.image_scale"].operator float();
    rendered_image_viewer_scale_main_ =
        settings_file["GaussianViewer.image_scale_main"].operator float();

    // Localization
    loc_camera_cfg_path =
        settings_file["Localization.camera_cfg_path"].operator std::string();
}

void GaussianMapper::run()
{
    std::cout << "!!! GaussianMapper::run" << std::endl;
    // First loop: Initial gaussian mapping
    while (!isStopped()) {
        // Check conditions for initial mapping
        if (hasMetInitialMappingConditions()) {
            // Get initial map
            std::vector<float> new_points;
            std::vector<float> new_colors;
            std::vector<float> new_scales;
            std::vector<float> new_rots;

            {   // Locked block
                boost::unique_lock<boost::mutex> allKFLock(pSLAM_->frozenMapMutex);
                std::vector<dso::FrozenFrameHessian*>* keyframesFromDso = &pSLAM_->newframeHessians;

                int j = 0;
                int num_kf = keyframesFromDso->size();
                for (int idx=0; idx < num_kf; idx++) {
                    // dso::FrozenFrameHessian* fh = keyframesFromDso.at(idx);
                    dso::FrozenFrameHessian* fh = keyframesFromDso->front();

                    if (fh->pointsInWorld.empty())
                        break;

                    float fx = pSLAM_->Hcalib.fxl();
                    float fy = pSLAM_->Hcalib.fyl();
                    float cx = pSLAM_->Hcalib.cxl();
                    float cy = pSLAM_->Hcalib.cyl();

                    Sophus::SE3d c2w = fh->camToWorld;
                    
                    // for (dso::PointHessian* ph : fh->pointHessians) {
                    //     if (ph->idepth > 0) {
                    //         float depth = 1.0f / ph->idepth;
                    //         float x = ((ph->u - cx) / fx) * depth;
                    //         float y = ((ph->v - cy) / fy) * depth;
                    //         float z = depth;

                    //         // TODO: change to matrix calculation
                    //         Eigen::Vector3d point_in_camera(x, y, z);
                    //         Eigen::Vector3d point_in_world = c2w * point_in_camera;

                    //         Point3D point3D;
                    //         point3D.xyz_ = point_in_world;

                    //         point3D.color_(0) = ph->kf_color.at(0);
                    //         point3D.color_(1) = ph->kf_color.at(1);
                    //         point3D.color_(2) = ph->kf_color.at(2);

                    //         scene_->cachePoint3D(j, point3D);
                    //     }
                    //     ++j;
                    // }

                    new_points.reserve(new_points.size() + fh->pointsInWorld.size());
                    new_points.insert(new_points.end(), fh->pointsInWorld.begin(), fh->pointsInWorld.end());

                    new_colors.reserve(new_colors.size() + fh->colors.size());
                    new_colors.insert(new_colors.end(), fh->colors.begin(), fh->colors.end());

                    new_scales.reserve(new_scales.size() + fh->scales.size());
                    new_scales.insert(new_scales.end(), fh->scales.begin(), fh->scales.end());

                    new_rots.reserve(new_rots.size() + fh->rots.size());
                    new_rots.insert(new_rots.end(), fh->rots.begin(), fh->rots.end());

                    std::shared_ptr<GaussianKeyframe> new_kf = std::make_shared<GaussianKeyframe>(fh->incomingID, getIteration());
                    // std::shared_ptr<GaussianKeyframe> new_kf = std::make_shared<GaussianKeyframe>(scene_->keyframes().size(), getIteration());
                    new_kf->zfar_ = z_far_;
                    new_kf->znear_ = z_near_;
                    // Pose
                    // auto pose = pKF->GetPose();
                    c2w = c2w.inverse();
                    new_kf->setPose(
                        // pose.unit_quaternion().cast<double>(),
                        // pose.translation().cast<double>()
                        c2w.unit_quaternion(),
                        c2w.translation());
                    cv::Mat imgRGB_undistorted, imgAux_undistorted;

                    try {
                        // Add first keyframe to the scene
                        Camera& camera = scene_->cameras_.at(0);
                        new_kf->setCameraParams(camera);

                        // Image (left if STEREO)
                        dso::MinimalImageB3* img = fh->kfImg;
                        cv::Mat imgRGB(img->h, img->w, CV_8UC3, img->data);
                        if (need_distortion) {
                            camera.undistortImage(imgRGB, imgRGB_undistorted);
                        } else {
                            imgRGB_undistorted = imgRGB;
                        }
                        // imgRGB_undistorted = imgRGB;

                        imgAux_undistorted = imgRGB_undistorted;
                        imgRGB_undistorted.convertTo(imgRGB_undistorted, CV_32FC3, 1.0 / 255.0);
                        new_kf->original_image_ =
                            tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
                        
                        // Depth image
                        // dso::MinimalImage<unsigned short>* depth_img = fh->kf_depth;
                        // cv::Mat imgDepth(depth_img->h, depth_img->w, CV_16UC1, depth_img->data);
                        // cv::Mat imgDepthFloat;
                        // imgDepth.convertTo(imgDepthFloat, CV_32FC1, 1.0 / depth_scale);
                        // new_kf->original_depth_ =
                        //     tensor_utils::cvMat2TorchTensor_Float32(imgDepthFloat, device_type_);

                        // Sparse Depth
                        new_kf->sparse_depth_ =
                            tensor_utils::cvMat2TorchTensor_Float32(fh->kfSparseDepth, device_type_);

                        // TODO: add if needed
                        new_kf->img_filename_ = result_dir_ / "img" / (std::to_string(fh->incomingID) + ".png");
                        new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
                        new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
                        new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
                    }
                    catch (std::out_of_range) {
                        throw std::runtime_error("[GaussianMapper::run]KeyFrame Camera not found!");
                    }

                    new_kf->computeTransformTensors();
                    scene_->addKeyframe(new_kf, &kfid_shuffled_);
                    increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

                    // Features
                    // std::vector<float> pixels;
                    // std::vector<float> pointsLocal;
                    // pKF->GetKeypointInfo(pixels, pointsLocal);
                    // new_kf->kps_pixel_ = std::move(pixels);
                    // new_kf->kps_point_local_ = std::move(pointsLocal);
                    new_kf->img_undist_ = imgRGB_undistorted;
                    new_kf->img_auxiliary_undist_ = imgAux_undistorted;
                    
                    if (inserted_kf_cursor < fh->incomingID)
                        inserted_kf_cursor = fh->incomingID;

                    keyframesFromDso->erase(keyframesFromDso->begin());
                }
            }   // Lock resolved

            // Prepare multi resolution images for training
            for (auto& kfit : scene_->keyframes()) {
                auto pkf = kfit.second;
                if (device_type_ == torch::kCUDA) {
                    cv::cuda::GpuMat img_gpu;
                    img_gpu.upload(pkf->img_undist_);
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::cuda::GpuMat img_resized;
                        cv::cuda::resize(img_gpu, img_resized,
                                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
                    }
                }
                else {
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::Mat img_resized;
                        cv::resize(pkf->img_undist_, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
                    }
                }
            }
            
            // Prepare for training
            {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
                // gaussians_->createFromPcd(new_points, new_colors, scene_->cameras_extent_);
                gaussians_->createFromPcd(new_points, new_colors, new_scales, new_rots, scene_->cameras_extent_);
                std::unique_lock<std::mutex> lock(mutex_settings_);
                gaussians_->trainingSetup(opt_params_);
            }
            // Invoke training once
            trainForOneIteration();
            // Finish initial mapping loop
            initial_mapped_ = true;
            break;
        }
        // else if (pSLAM_->isRunMapping()) {
        //     break;
        // }
        else {
            // Initial conditions not satisfied
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Second loop: Incremental gaussian mapping
    std::cout << "start training loop" << std::endl;
    int SLAM_stop_iter = 0;
    while (!isStopped()) {
        // Update DSO keyframes
        if (checkKFupdateFromGSRequested()) {
            // std::cout << "Keyframe Update Requested" << std::endl;
            if (scene_->keyframes_.size() > 10) {
                updateKeyFramesFromGS();
            }
            // std::cout << "Update End" << std::endl;
            pSLAM_->isDoneKFUpdateFromGS = true;
        }

        // Check conditions for incremental mapping
        if (hasMetIncrementalMappingConditions()) {
            combineMappingOperations();
            // if (cull_keyframes_)
            //     cullKeyframes();
        }

        // Invoke training once
        trainForOneIteration();

        if (!SLAM_ended_ && !pSLAM_->isRunMapping()) {
            SLAM_stop_iter = getIteration();
            SLAM_ended_ = true;
            // ablation study!!!!!!!!!!!
            // additional_training_start = std::chrono::steady_clock::now();
            // additional_training_start_iter = getIteration();
        }

        
        if (SLAM_ended_ || getIteration() >= opt_params_.iterations_)
            break;

        // ablation study!!!!!!!!!!!
        // if (getIteration() - additional_training_start_iter > 13000)
        //     break;

        // if (getIteration() >= opt_params_.iterations_)
        //     break;
    }

    // Third loop: Tail gaussian optimization
    // int densify_interval = densifyInterval();
    // int n_delay_iters = densify_interval * 0.8;
    // while (getIteration() - SLAM_stop_iter <= n_delay_iters || getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) {
    //     trainForOneIteration();
    //     densify_interval = densifyInterval();
    //     n_delay_iters = densify_interval * 0.8;
    // }

    // Save and clear
    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / "gs_map");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    signalStop();
}

void GaussianMapper::trainColmap()
{
    // Prepare multi resolution images for training
    for (auto& kfit : scene_->keyframes()) {
        auto pkf = kfit.second;
        increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());
        if (device_type_ == torch::kCUDA) {
            cv::cuda::GpuMat img_gpu;
            img_gpu.upload(pkf->img_undist_);
            pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::cuda::GpuMat img_resized;
                cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                pkf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
            }
        }
        else {
            pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::Mat img_resized;
                cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                pkf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
            }
        }
    }

    // Prepare for training
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
        gaussians_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
        std::unique_lock<std::mutex> lock(mutex_settings_);
        gaussians_->trainingSetup(opt_params_);
        this->initial_mapped_ = true;
    }

    // Main loop: gaussian splatting training
    while (!isStopped()) {
        // Invoke training once
        trainForOneIteration();

        if (getIteration() >= opt_params_.iterations_)
            break;
    }

    // Tail gaussian optimization
    int densify_interval = densifyInterval();
    int n_delay_iters = densify_interval * 0.8;
    while (getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) {
        trainForOneIteration();
        densify_interval = densifyInterval();
        n_delay_iters = densify_interval * 0.8;
    }

    // Save and clear
    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    signalStop();
}

/**
 * @brief The training iteration body
 * 
 */
void GaussianMapper::trainForOneIteration()
{
    increaseIteration(1);
    auto iter_start_timing = std::chrono::steady_clock::now();

    // Pick a random Camera
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    // std::shared_ptr<GaussianKeyframe> viewpoint_cam = useOneRandomKeyframe();
    if (!viewpoint_cam) {
        increaseIteration(-1);
        return;
    }

    // writeKeyframeUsedTimes(result_dir_ / "used_times");

    // if (isdoingInactiveGeoDensify() && !viewpoint_cam->done_inactive_geo_densify_)
    //     increasePcdByKeyframeInactiveGeoDensify(viewpoint_cam);

    int training_level = num_gaus_pyramid_sub_levels_;
    int image_height, image_width;
    torch::Tensor gt_image, mask, gt_depth, sparse_depth;
    if (isdoingGausPyramidTraining())
        training_level = viewpoint_cam->getCurrentGausPyramidLevel();
    if (training_level == num_gaus_pyramid_sub_levels_) {
        image_height = viewpoint_cam->image_height_;
        image_width = viewpoint_cam->image_width_;
        gt_image = viewpoint_cam->original_image_.cuda();
        mask = undistort_mask_[viewpoint_cam->camera_id_];

        // gt_depth = viewpoint_cam->original_depth_.cuda();
        sparse_depth = viewpoint_cam->sparse_depth_.cuda();
    }
    else {
        image_height = viewpoint_cam->gaus_pyramid_height_[training_level];
        image_width = viewpoint_cam->gaus_pyramid_width_[training_level];
        gt_image = viewpoint_cam->gaus_pyramid_original_image_[training_level].cuda();
        mask = scene_->cameras_.at(viewpoint_cam->camera_id_).gaus_pyramid_undistort_mask_[training_level];
    }

    // Mutex lock for usage of the gaussian model
    std::unique_lock<std::mutex> lock_render(mutex_render_);

    // Every 1000 its we increase the levels of SH up to a maximum degree
    if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
        default_sh_ += 1;
    // if (isdoingGausPyramidTraining())
    //     gaussians_->setShDegree(training_level);
    // else
        gaussians_->setShDegree(default_sh_);

    // Update learning rate
    if (pSLAM_) {
        int used_times = kfs_used_times_[viewpoint_cam->fid_];
        int step = (used_times <= opt_params_.position_lr_max_steps_ ? used_times : opt_params_.position_lr_max_steps_);
        float position_lr = gaussians_->updateLearningRate(step);
        setPositionLearningRateInit(position_lr);
    }
    else {
        gaussians_->updateLearningRate(getIteration());
    }

    gaussians_->setFeatureLearningRate(featureLearningRate());
    gaussians_->setOpacityLearningRate(opacityLearningRate());
    gaussians_->setScalingLearningRate(scalingLearningRate());
    gaussians_->setRotationLearningRate(rotationLearningRate());

    // Render
    // Outputs:
    // rgb_img, viewspace_points, visibility_filter, radii, render_alpha, render_normal, render_dist, surf_depth, surf_normal
    auto render_pkg = GaussianRenderer::render(
        viewpoint_cam,
        image_height,
        image_width,
        gaussians_,
        pipe_params_,
        background_,
        override_color_
    );
    auto rendered_image = std::get<0>(render_pkg);
    auto viewspace_point_tensor = std::get<1>(render_pkg);
    auto visibility_filter = std::get<2>(render_pkg);
    auto radii = std::get<3>(render_pkg);
    // 2DGS
    // auto rend_alpha = std::get<4>(render_pkg);
    auto rend_normal = std::get<5>(render_pkg);
    auto rend_dist = std::get<6>(render_pkg);
    auto surf_depth = std::get<7>(render_pkg);
    auto surf_normal = std::get<8>(render_pkg);
    auto median_depth = std::get<9>(render_pkg);

    // Get rid of black edges caused by undistortion
    torch::Tensor masked_image = rendered_image * mask;

    // Loss
    // original Gaussian Splatting
    auto Ll1 = loss_utils::l1_loss(masked_image, gt_image);
    float lambda_dssim = lambdaDssim();

    // 2DGS
    float lambda_normal = 0.01f;
    float lambda_dist = 0.0f;
    auto normal_error = (1 - (rend_normal * surf_normal)).sum(0).unsqueeze(0);
    auto normal_loss = (normal_error).mean();
    // auto dist_loss = (rend_dist).mean();

    auto loss = (1.0 - lambda_dssim) * Ll1
            + lambda_dssim * (1.0 - loss_utils::ssim(masked_image, gt_image, device_type_))
            + lambda_normal * normal_loss;
            // + lambda_dist * dist_loss;

    if (training_level == num_gaus_pyramid_sub_levels_) {
        torch::Tensor masked_depth = surf_depth * mask;
        // auto depth_L1 = loss_utils::l1_loss(masked_depth, gt_depth);
        // loss += 0.5 * depth_L1;

        auto sparse_depth_nonzero_mask = (sparse_depth != 0).to(masked_depth.dtype());
        auto valid_masked_depth = masked_depth * sparse_depth_nonzero_mask;
        auto valid_sparse_depth = sparse_depth * sparse_depth_nonzero_mask;
        auto depth_L1 = torch::abs(valid_masked_depth - valid_sparse_depth).sum() / sparse_depth_nonzero_mask.sum();
        loss += lambda_sparse_depth * depth_L1;
    }

    loss.backward();

    torch::cuda::synchronize();
    {
        torch::NoGradGuard no_grad;
        ema_loss_for_log_ = 0.4f * loss.item().toFloat() + 0.6 * ema_loss_for_log_;

        // std::cout << getIteration() << std::endl;

        if (keyframe_record_interval_ &&
            getIteration() % keyframe_record_interval_ == 0)
            recordKeyframeRendered(masked_image, gt_image, surf_depth, viewpoint_cam->fid_, result_dir_, result_dir_, result_dir_, result_dir_);

        // Densification
        if (getIteration() < opt_params_.densify_until_iter_) {
            // Keep track of max radii in image-space for pruning
            gaussians_->max_radii2D_.index_put_(
                {visibility_filter},
                torch::max(gaussians_->max_radii2D_.index({visibility_filter}),
                            radii.index({visibility_filter})));
            // if (!isdoingGausPyramidTraining() || training_level < num_gaus_pyramid_sub_levels_)
                gaussians_->addDensificationStats(viewspace_point_tensor, visibility_filter);

            if ((getIteration() > opt_params_.densify_from_iter_) &&
                (getIteration() % densifyInterval()== 0)) {
                int size_threshold = (getIteration() > prune_big_point_after_iter_) ? 2000000 : 0;   // 20
                gaussians_->densifyAndPrune(
                    densifyGradThreshold(),
                    densify_min_opacity_,//0.005,//
                    scene_->cameras_extent_,
                    size_threshold
                );
            }

            if (opacityResetInterval()
                && (getIteration() % opacityResetInterval() == 0
                    ||(model_params_.white_background_ && getIteration() == opt_params_.densify_from_iter_)))
                gaussians_->resetOpacity();
        }

        auto iter_end_timing = std::chrono::steady_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        iter_end_timing - iter_start_timing).count();

        // Log and save
        if (training_report_interval_ && (getIteration() % training_report_interval_ == 0))
            GaussianTrainer::trainingReport(
                getIteration(),
                opt_params_.iterations_,
                Ll1,
                loss,
                ema_loss_for_log_,
                loss_utils::l1_loss,
                iter_time,
                *gaussians_,
                *scene_,
                pipe_params_,
                background_
            );
        if ((all_keyframes_record_interval_ && getIteration() % all_keyframes_record_interval_ == 0)
            // || loop_closure_iteration_
            )
        {
            renderAndRecordAllKeyframes();
            savePly(result_dir_ / std::to_string(getIteration()) / "ply");
        }

        // ablation study!!!!!!!!!!!
        // if (SLAM_ended_ && (getIteration() - additional_training_start_iter) % 100 == 0)
        // {
        //     int additional_training_iter = getIteration() - additional_training_start_iter;
        //     std::chrono::steady_clock::time_point start_saving = std::chrono::steady_clock::now();
        //     double duration = std::chrono::duration_cast<std::chrono::duration<double>>(start_saving - additional_training_start).count();
        //     std::ofstream outFile(result_dir_ / "additional_training_time.txt", std::ios::app);

        //     // outFile << "saving " << std::to_string(additional_training_iter) << " iteration time: " << duration << std::endl;
        //     outFile << duration << std::endl;
        //     savePly(result_dir_ / ("additional_" + std::to_string(additional_training_iter)) / "ply");

        //     std::chrono::steady_clock::time_point end_saving = std::chrono::steady_clock::now();
        //     duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_saving - additional_training_start).count();
        //     // outFile << "saving end" << std::to_string(additional_training_iter) << " iteration time: " << duration << std::endl;
        //     outFile << duration << std::endl;

        //     outFile.close();
        // }

        if (loop_closure_iteration_)
            loop_closure_iteration_ = false;

        // Optimizer step
        if (getIteration() < opt_params_.iterations_) {
            gaussians_->optimizer_->step();
            gaussians_->optimizer_->zero_grad(true);
        }
    }
}

bool GaussianMapper::isStopped()
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    return this->stopped_;
}

void GaussianMapper::signalStop(const bool going_to_stop)
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    this->stopped_ = going_to_stop;
}

bool GaussianMapper::hasMetInitialMappingConditions()
{
    if (checkKFupdateFromGSRequested()) {
        pSLAM_->isDoneKFUpdateFromGS = true;
    }
    // if (pSLAM_->allKeyframeHessians.size() >= min_num_initial_map_kfs_)
    if (pSLAM_->initialized && pSLAM_->newframeHessians.size() >= min_num_initial_map_kfs_)
        return true;

    return false;
}

bool GaussianMapper::hasMetIncrementalMappingConditions()
{
    // if (inserted_kf_cursor < pSLAM_->lastKFIncomingId)
    if (pSLAM_->newframeHessians.size() > 0)
        return true;

    // inserted_kf_cursor

    return false;
}

bool GaussianMapper::checkKFupdateFromGSRequested()
{
    return pSLAM_->callKFUpdateFromGS;
}

void GaussianMapper::combineMappingOperations()
{
    std::vector<float> new_points;
    std::vector<float> new_colors;
    std::vector<float> new_scales;
    std::vector<float> new_rots;

    {
        boost::unique_lock<boost::mutex> allKFLock(pSLAM_->frozenMapMutex);
        std::vector<dso::FrozenFrameHessian*>* keyframesFromDso = &pSLAM_->newframeHessians;

        // while (keyframesFromDso.size() > 0) {
        int num_kf = keyframesFromDso->size();
        for (int idx=0; idx < num_kf; idx++) {
            // dso::FrozenFrameHessian* fh = keyframesFromDso.at(idx);
            dso::FrozenFrameHessian* fh = keyframesFromDso->front();
            
            if ( fh->pointsInWorld.empty())
                continue;

            std::cout << "New KF from DSO to 2DGS, incoming_id: " << fh->incomingID << std::endl;

            float fx = pSLAM_->Hcalib.fxl();
            float fy = pSLAM_->Hcalib.fyl();
            float cx = pSLAM_->Hcalib.cxl();
            float cy = pSLAM_->Hcalib.cyl();

            Sophus::SE3d c2w = fh->camToWorld;

            // int j = 0;
            // for (dso::PointHessian* ph : fh->pointHessians) {
            //     if (ph->idepth > 0) {
            //         float depth = 1.0f / ph->idepth;
            //         float x = ((ph->u - cx) / fx) * depth;
            //         float y = ((ph->v - cy) / fy) * depth;
            //         float z = depth;

            //         Eigen::Vector3d point_in_camera(x, y, z);
            //         Eigen::Vector3d point_in_world = c2w * point_in_camera;

            //         new_points.push_back(point_in_world.x());
            //         new_points.push_back(point_in_world.y());
            //         new_points.push_back(point_in_world.z());

            //         new_colors.push_back(ph->kf_color.at(0));
            //         new_colors.push_back(ph->kf_color.at(1));
            //         new_colors.push_back(ph->kf_color.at(2));
            //     }
            //     ++j;
            // }

            // get initial GS from DSO
            new_points.reserve(new_points.size() + fh->pointsInWorld.size());
            new_points.insert(new_points.end(), fh->pointsInWorld.begin(), fh->pointsInWorld.end());

            new_colors.reserve(new_colors.size() + fh->colors.size());
            new_colors.insert(new_colors.end(), fh->colors.begin(), fh->colors.end());

            new_scales.reserve(new_scales.size() + fh->scales.size());
            new_scales.insert(new_scales.end(), fh->scales.begin(), fh->scales.end());

            new_rots.reserve(new_rots.size() + fh->rots.size());
            new_rots.insert(new_rots.end(), fh->rots.begin(), fh->rots.end());

            // std::cout << "Num_points " << new_points.size()/3 << " Points" << std::endl;

            std::shared_ptr<GaussianKeyframe> new_kf = std::make_shared<GaussianKeyframe>(fh->incomingID, getIteration());
            new_kf->zfar_ = z_far_;
            new_kf->znear_ = z_near_;
            // Pose
            // auto pose = pKF->GetPose();
            // Sophus::SE3d c2w = fh->shell->camToWorld; declared before
            c2w = c2w.inverse();
            new_kf->setPose(
                // pose.unit_quaternion().cast<double>(),
                // pose.translation().cast<double>()
                c2w.unit_quaternion(),
                c2w.translation());
            cv::Mat imgRGB_undistorted, imgAux_undistorted;
            try {
                // Add first keyframe to the scene
                Camera& camera = scene_->cameras_.at(0);
                new_kf->setCameraParams(camera);

                // Image (left if STEREO)
                dso::MinimalImageB3* img = fh->kfImg;
                cv::Mat imgRGB(img->h, img->w, CV_8UC3, img->data);

                if (need_distortion) {
                    camera.undistortImage(imgRGB, imgRGB_undistorted);
                } else {
                    imgRGB_undistorted = imgRGB;
                }
                // camera.undistortImage(imgRGB, imgRGB_undistorted);
                // imgRGB_undistorted = imgRGB;

                imgAux_undistorted = imgRGB_undistorted;
                imgRGB_undistorted.convertTo(imgRGB_undistorted, CV_32FC3, 1.0 / 255.0);
                new_kf->original_image_ =
                    tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);

                // Depth image
                // dso::MinimalImage<unsigned short>* depth_img = fh->kf_depth;
                // cv::Mat imgDepth(depth_img->h, depth_img->w, CV_16UC1, depth_img->data);
                // cv::Mat imgDepthFloat;
                // imgDepth.convertTo(imgDepthFloat, CV_32FC1, 1.0 / depth_scale);
                // new_kf->original_depth_ =
                //     tensor_utils::cvMat2TorchTensor_Float32(imgDepthFloat, device_type_);

                // Sparse Depth
                new_kf->sparse_depth_ =
                    tensor_utils::cvMat2TorchTensor_Float32(fh->kfSparseDepth, device_type_);

                new_kf->img_filename_ = result_dir_ / "img" / (std::to_string(fh->incomingID) + ".png");
                new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
                new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
                new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
            }
            catch (std::out_of_range) {
                throw std::runtime_error("[GaussianMapper::run]KeyFrame Camera not found!");
            }
            new_kf->computeTransformTensors();
            scene_->addKeyframe(new_kf, &kfid_shuffled_);
            increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());
            // Features
            // std::vector<float> pixels;
            // std::vector<float> pointsLocal;
            // pKF->GetKeypointInfo(pixels, pointsLocal);
            // new_kf->kps_pixel_ = std::move(pixels);
            // new_kf->kps_point_local_ = std::move(pointsLocal);
            new_kf->img_undist_ = imgRGB_undistorted;
            new_kf->img_auxiliary_undist_ = imgAux_undistorted;
            // Prepare multi resolution images for training
            if (device_type_ == torch::kCUDA) {
                cv::cuda::GpuMat img_gpu;
                img_gpu.upload(new_kf->img_undist_);
                new_kf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                    cv::cuda::GpuMat img_resized;
                    cv::cuda::resize(img_gpu, img_resized,
                                        cv::Size(new_kf->gaus_pyramid_width_[l], new_kf->gaus_pyramid_height_[l]));
                    new_kf->gaus_pyramid_original_image_[l] =
                        tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
                }
            }
            else {
                new_kf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                    cv::Mat img_resized;
                    cv::resize(new_kf->img_undist_, img_resized,
                                cv::Size(new_kf->gaus_pyramid_width_[l], new_kf->gaus_pyramid_height_[l]));
                    new_kf->gaus_pyramid_original_image_[l] =
                        tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
                }
            }

            inserted_kf_cursor = fh->incomingID;

            keyframesFromDso->erase(keyframesFromDso->begin());
        }
    }

    // Add new gaussians
    if (!new_points.empty() and initial_mapped_) {
        torch::NoGradGuard no_grad;
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        // gaussians_->increasePcd(new_points, new_colors, getIteration());
        gaussians_->increasePcd(new_points, new_colors, new_scales, new_rots, getIteration());
        // std::cout << "Inserted " << new_points.size()/3 << " Points to the GS Scene" << std::endl;
        // std::cout << "add new points to the scene" << std::endl;
    }

    updateGSKeyFramesFromDSO();

    // if (scene_->keyframes_.size() > 10)
    //     updateKeyFramesFromGS();
}

std::vector<std::tuple<torch::Tensor, torch::Tensor>> GaussianMapper::renderKFDepths(std::vector<Sophus::SE3d> kf_poses)
{
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> depth_maps_out;

    // render depth imgs
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    for (Sophus::SE3d kfPose : kf_poses)
    {
        kfPose = kfPose.inverse();
        std::tuple<torch::Tensor, torch::Tensor> renderedDepth = renderDepthFromPose(kfPose, image_width, image_height, false);
        depth_maps_out.push_back(renderedDepth);
    }

    return depth_maps_out;
}

void GaussianMapper::handleNewKeyframe(
    std::tuple< unsigned long/*Id*/,
                unsigned long/*CameraId*/,
                Sophus::SE3f/*pose*/,
                cv::Mat/*image*/,
                bool/*isLoopClosure*/,
                cv::Mat/*auxiliaryImage*/,
                std::vector<float>,
                std::vector<float>,
                std::string> &kf)
{
    std::shared_ptr<GaussianKeyframe> pkf =
        std::make_shared<GaussianKeyframe>(std::get<0>(kf), getIteration());
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    auto& pose = std::get<2>(kf);
    pkf->setPose(
        pose.unit_quaternion().cast<double>(),
        pose.translation().cast<double>());
    cv::Mat imgRGB_undistorted, imgAux_undistorted;
    try {
        // Camera
        Camera& camera = scene_->cameras_.at(std::get<1>(kf));
        pkf->setCameraParams(camera);

        // Image (left if STEREO)
        cv::Mat imgRGB = std::get<3>(kf);
        if (this->sensor_type_ == STEREO)
            imgRGB_undistorted = imgRGB;
        else
            camera.undistortImage(imgRGB, imgRGB_undistorted);
            
        // Auxiliary Image
        cv::Mat imgAux = std::get<5>(kf);
        if (this->sensor_type_ == RGBD)
            camera.undistortImage(imgAux, imgAux_undistorted);
        else
            imgAux_undistorted = imgAux;

        pkf->original_image_ =
            tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
        pkf->img_filename_ = std::get<8>(kf);
        pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
        pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
        pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::combineMappingOperations]KeyFrame Camera not found!");
    }
    // Add the new keyframe to the scene
    pkf->computeTransformTensors();
    scene_->addKeyframe(pkf, &kfid_shuffled_);

    // Give new keyframes times of use and add it to the training sliding window
    increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

    // Get dense point cloud from the new keyframe to accelerate training
    pkf->img_undist_ = imgRGB_undistorted;
    pkf->img_auxiliary_undist_ = imgAux_undistorted;
    pkf->kps_pixel_ = std::move(std::get<6>(kf));
    pkf->kps_point_local_ = std::move(std::get<7>(kf));
    if (isdoingInactiveGeoDensify())
        increasePcdByKeyframeInactiveGeoDensify(pkf);

    // Prepare multi resolution images for training
    if (device_type_ == torch::kCUDA) {
        cv::cuda::GpuMat img_gpu;
        img_gpu.upload(pkf->img_undist_);
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::cuda::GpuMat img_resized;
            cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
        }
    }
    else {
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::Mat img_resized;
            cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
        }
    }
}

void GaussianMapper::generateKfidRandomShuffle()
{

    // std::cout << "KFs: " << kfid_in_dso_active_window << std::endl;

    if (scene_->keyframes().empty())
        return;

    std::size_t nkfs = scene_->keyframes().size();
    kfid_shuffle_.resize(nkfs);
    std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
    std::mt19937 g(rd_());
    std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    // std::shuffle(kfid_in_dso_active_window.begin(), kfid_in_dso_active_window.end(), g);

    // std::cout << "Shuffled: " << kfid_in_dso_active_window << std::endl;

    kfid_training = kfid_in_dso_active_window.size() - 1;
    kfid_shuffle_idx_ = 0;
    kfid_shuffled_ = true;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomSlidingWindowKeyframe()
{
// auto t1 = std::chrono::steady_clock::now();
    if (scene_->keyframes().empty() || kfid_in_dso_active_window.empty())
        return nullptr;

    if (!kfid_shuffled_)
        generateKfidRandomShuffle();

    std::shared_ptr<GaussianKeyframe> viewpoint_cam = nullptr;
    int random_cam_idx, random_cam_fid;

    // original
    if (kfid_shuffled_) {
        int start_shuffle_idx = kfid_shuffle_idx_;
        do {
            // Next shuffled idx
            ++kfid_shuffle_idx_;
            if (kfid_shuffle_idx_ >= kfid_shuffle_.size())
                kfid_shuffle_idx_ = 0;
            // Add 1 time of use to all kfs if they are all unavalible
            if (kfid_shuffle_idx_ == start_shuffle_idx)
                for (auto& kfit : scene_->keyframes())
                    increaseKeyframeTimesOfUse(kfit.second, 1);
            // Get viewpoint kf
            random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];
            auto random_cam_it = scene_->keyframes().begin();
            for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
                ++random_cam_it;
            viewpoint_cam = (*random_cam_it).second;
        } while (viewpoint_cam->remaining_times_of_use_ <= 0);
    }

    // mine 2
    // if (kfid_shuffled_) {
    //     if (kfid_in_dso_active_window.size() <= kfid_training) {
    //         if (kfid_shuffle_idx_ >= kfid_shuffle_.size())
    //             kfid_shuffle_idx_ = 0;
    //         random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];
    //         auto random_cam_it = scene_->keyframes().begin();
    //         for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
    //             ++random_cam_it;
    //         viewpoint_cam = (*random_cam_it).second;
    //         std::cout << "Random KF" << std::endl;
    //         ++kfid_shuffle_idx_;
    //     } else {
    //         random_cam_fid = kfid_in_dso_active_window[kfid_training];

    //         bool use_active_window = false;
    //         for (int idx = kfid_training; idx >= 0; idx--) {
    //             if (kfs_used_times_.find(random_cam_fid) == kfs_used_times_.end()) {
    //                 // this keyframe is not used before
    //                 viewpoint_cam = scene_->getKeyframe(random_cam_fid);
    //                 kfid_training = idx;
    //                 std::cout << "Active window" << std::endl;
    //                 use_active_window = true;
    //                 break;
    //             } else {
    //                 if (kfs_used_times_[random_cam_fid] < 4) {
    //                     viewpoint_cam = scene_->getKeyframe(random_cam_fid);
    //                     kfid_training = idx;
    //                     std::cout << "Active window" << std::endl;
    //                     use_active_window = true;
    //                     break;
    //                 }
    //             }
    //         }

    //         if (!use_active_window) {
    //             if (kfid_shuffle_idx_ >= kfid_shuffle_.size())
    //                 kfid_shuffle_idx_ = 0;
    //             random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];
    //             auto random_cam_it = scene_->keyframes().begin();
    //             for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
    //                 ++random_cam_it;
    //             viewpoint_cam = (*random_cam_it).second;
    //             std::cout << "Random KF" << std::endl;
    //             ++kfid_shuffle_idx_;
    //         }
    //     }
    // }

    // mine 3


    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];
    
    // Handle times of use
    --(viewpoint_cam->remaining_times_of_use_);

// auto t2 = std::chrono::steady_clock::now();
// auto t21 = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
// std::cout<<t21 <<" ns"<<std::endl;
    return viewpoint_cam;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomKeyframe()
{
    if (scene_->keyframes().empty())
        return nullptr;

    // Get randomly
    int nkfs = static_cast<int>(scene_->keyframes().size());
    int random_cam_idx = std::rand() / ((RAND_MAX + 1u) / nkfs);
    auto random_cam_it = scene_->keyframes().begin();
    for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
        ++random_cam_it;
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = (*random_cam_it).second;

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];

    --(viewpoint_cam->remaining_times_of_use_);

    return viewpoint_cam;
}

void GaussianMapper::increaseKeyframeTimesOfUse(
    std::shared_ptr<GaussianKeyframe> pkf,
    int times)
{
    pkf->remaining_times_of_use_ += times;
}

void GaussianMapper::cullKeyframes()
{
    // TODO
    // std::unordered_set<unsigned long> kfids =
    //     pSLAM_->getAtlas()->GetCurrentKeyFrameIds();
    // std::vector<unsigned long> kfids_to_erase;
    // std::size_t nkfs = scene_->keyframes().size();
    // kfids_to_erase.reserve(nkfs);
    // for (auto& kfit : scene_->keyframes()) {
    //     unsigned long kfid = kfit.first;
    //     if (kfids.find(kfid) == kfids.end()) {
    //         kfids_to_erase.emplace_back(kfid);
    //     }
    // }

    // for (auto& kfid : kfids_to_erase) {
    //     scene_->keyframes().erase(kfid);
    // }
}

void GaussianMapper::increasePcdByKeyframeInactiveGeoDensify(
    std::shared_ptr<GaussianKeyframe> pkf)
{
// auto start_timing = std::chrono::steady_clock::now();
    torch::NoGradGuard no_grad;

    Sophus::SE3f Twc = pkf->getPosef().inverse();

    switch (this->sensor_type_)
    {
    case MONOCULAR:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        assert(pkf->kps_pixel_.size() % 2 == 0);
        int N = pkf->kps_pixel_.size() / 2;
        torch::Tensor kps_pixel_tensor = torch::from_blob(
            pkf->kps_pixel_.data(), {N, 2},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        torch::Tensor kps_point_local_tensor = torch::from_blob(
            pkf->kps_point_local_.data(), {N, 3},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        torch::Tensor kps_has3D_tensor = torch::where(
            kps_point_local_tensor.index({torch::indexing::Slice(), 2}) > 0.0f, true, false);

        cv::cuda::GpuMat rgb_gpu;
        rgb_gpu.upload(pkf->img_undist_);
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

        auto result =
            monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
                kps_pixel_tensor, kps_has3D_tensor, kps_point_local_tensor, colors,
                monocular_inactive_geo_densify_max_pixel_dist_, pkf->intr_, pkf->image_width_);
        torch::Tensor& points3D_valid = std::get<0>(result);
        torch::Tensor& colors_valid = std::get<1>(result);
        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);
        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    case STEREO:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        cv::cuda::GpuMat rgb_left_gpu, rgb_right_gpu;
        cv::cuda::GpuMat gray_left_gpu, gray_right_gpu;

        rgb_left_gpu.upload(pkf->img_undist_);
        rgb_right_gpu.upload(pkf->img_auxiliary_undist_);

        // From CV_32FC3 to CV_32FC1
        cv::cuda::cvtColor(rgb_left_gpu, gray_left_gpu, cv::COLOR_RGB2GRAY);
        cv::cuda::cvtColor(rgb_right_gpu, gray_right_gpu, cv::COLOR_RGB2GRAY);

        // From CV_32FC1 to CV_8UC1
        gray_left_gpu.convertTo(gray_left_gpu, CV_8UC1, 255.0);
        gray_right_gpu.convertTo(gray_right_gpu, CV_8UC1, 255.0);

        // Compute disparity
        cv::cuda::GpuMat cv_disp;
        stereo_cv_sgm_->compute(gray_left_gpu, gray_right_gpu, cv_disp);
        cv_disp.convertTo(cv_disp, CV_32F, 1.0 / 16.0);

        // Reproject to get 3D points
        cv::cuda::GpuMat cv_points3D;
        cv::cuda::reprojectImageTo3D(cv_disp, cv_points3D, stereo_Q_, 3);

        // From cv::cuda::GpuMat to torch::Tensor
        torch::Tensor disp = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_disp);
        disp = disp.flatten(0, 1).contiguous();
        torch::Tensor points3D = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_points3D);
        points3D = points3D.permute({1, 2, 0}).flatten(0, 1).contiguous();
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_left_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();
    
        // Clear undisired and unreliable stereo points
        torch::Tensor point_valid_flags = torch::full(
            {disp.size(0)}, false, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
        int nkps_twice = pkf->kps_pixel_.size();
        int width = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]) + static_cast<int>(/*v*/pkf->kps_pixel_[kpidx + 1]) * width;
            // int u = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]);
            // if (u < 0.3 * width || u > 0.7 * width)
            point_valid_flags[idx] = true;
            // idx += width;
            // if (idx < disp.size(0)) {
            //     point_valid_flags[idx - 3] = true;
            //     point_valid_flags[idx - 2] = true;
            //     point_valid_flags[idx - 1] = true;
            //     point_valid_flags[idx] = true;
            // }
            // idx -= (2 * width);
            // if (idx > 0) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx + 1] = true;
            //     point_valid_flags[idx + 2] = true;
            //     point_valid_flags[idx + 3] = true;
            // }
            // idx += width;
            // idx += 3;
            // if (idx < disp.size(0)) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx - 1] = true;
            //     point_valid_flags[idx - 2] = true;
            // }
            // idx -= 6;
            // if (idx > 0) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx + 1] = true;
            //     point_valid_flags[idx + 2] = true;
            // }
        }
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(disp > static_cast<float>(stereo_cv_sgm_->getMinDisparity()), true, false));
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(disp < static_cast<float>(stereo_cv_sgm_->getNumDisparities()), true, false));

        torch::Tensor points3D_valid = points3D.index({point_valid_flags});
        torch::Tensor colors_valid = colors.index({point_valid_flags});

        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    case RGBD:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        cv::cuda::GpuMat img_rgb_gpu, img_depth_gpu;
        img_rgb_gpu.upload(pkf->img_undist_);
        img_depth_gpu.upload(pkf->img_auxiliary_undist_);

        // From cv::cuda::GpuMat to torch::Tensor
        torch::Tensor rgb = tensor_utils::cvGpuMat2TorchTensor_Float32(img_rgb_gpu);
        rgb = rgb.permute({1, 2, 0}).flatten(0, 1).contiguous();
        torch::Tensor depth = tensor_utils::cvGpuMat2TorchTensor_Float32(img_depth_gpu);
        depth = depth.flatten(0, 1).contiguous();

        // To clear undisired and unreliable depth
        torch::Tensor point_valid_flags = torch::full(
            {depth.size(0)}, false/*true*/, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
        int nkps_twice = pkf->kps_pixel_.size();
        int width = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]) + static_cast<int>(/*v*/pkf->kps_pixel_[kpidx + 1]) * width;
            point_valid_flags[idx] = true;
        }
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth > RGBD_min_depth_, true, false));
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth < RGBD_max_depth_, true, false));

        torch::Tensor colors_valid = rgb.index({point_valid_flags});

        // Reproject to get 3D points
        torch::Tensor points3D_valid;
        Camera& camera = scene_->cameras_.at(pkf->camera_id_);
        switch (camera.model_id_)
        {
        case Camera::PINHOLE:
        {
            points3D_valid = reprojectDepthPinhole(
                depth, point_valid_flags, pkf->intr_, pkf->image_width_);
        }
        break;
        case Camera::FISHEYE:
        {
            //TODO: support fisheye camera?
            throw std::runtime_error("[Gaussian Mapper]Fisheye cameras are not supported currently!");
        }
        break;
        default:
        {
            throw std::runtime_error("[Gaussian Mapper]Invalid camera model!");
        }
        break;
        }
        points3D_valid = points3D_valid.index({point_valid_flags});

        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    default:
    {
        throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }
    break;
    }

    pkf->done_inactive_geo_densify_ = true;
    ++depth_cached_;

    if (depth_cached_ >= max_depth_cached_) {
        depth_cached_ = 0;
        // Add new points to the model
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        gaussians_->increasePcd(depth_cache_points_, depth_cache_colors_, getIteration());
    }

// auto end_timing = std::chrono::steady_clock::now();
// auto completion_time = std::chrono::duration_cast<std::chrono::milliseconds>(
//                 end_timing - start_timing).count();
// std::cout << "[Gaussian Mapper]increasePcdByKeyframeInactiveGeoDensify() takes "
//             << completion_time
//             << " ms"
//             << std::endl;
}

// bool GaussianMapper::needInterruptTraining()
// {
//     std::unique_lock<std::mutex> lock_status(this->mutex_status_);
//     return this->interrupt_training_;
// }

// void GaussianMapper::setInterruptTraining(const bool interrupt_training)
// {
//     std::unique_lock<std::mutex> lock_status(this->mutex_status_);
//     this->interrupt_training_ = interrupt_training;
// }

void GaussianMapper::recordKeyframeRendered(
        torch::Tensor &rendered,
        torch::Tensor &ground_truth,
        torch::Tensor &depth_map,
        unsigned long kfid,
        std::filesystem::path result_img_dir,
        std::filesystem::path result_gt_dir,
        std::filesystem::path result_loss_dir,
        std::filesystem::path result_depth_dir,
        std::string name_suffix)
{
    if (record_rendered_image_) {
        auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered);
        // cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
        image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_img_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".jpg"), image_cv);
    }

    if (record_ground_truth_image_) {
        auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(ground_truth);
        // cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
        gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_gt_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_gt.jpg"), gt_image_cv);
    }

    if (record_loss_image_) {
        torch::Tensor loss_tensor = torch::abs(rendered - ground_truth);
        auto loss_image_cv = tensor_utils::torchTensor2CvMat_Float32(loss_tensor);
        // cv::cvtColor(loss_image_cv, loss_image_cv, CV_RGB2BGR);
        loss_image_cv.convertTo(loss_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_loss_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_loss.jpg"), loss_image_cv);
    }

    if (record_depth_image_) {
        depth_map = depth_map.squeeze(0);
        depth_map = depth_map.to(torch::kCPU);
        depth_map = depth_map.contiguous();
        // std::cout << depth_map.sizes() << std::endl;
        cv::Mat depth_cv(depth_map.size(0), depth_map.size(1), CV_32FC1, depth_map.data_ptr<float>());

        // auto depth_cv = tensor_utils::torchTensor2CvMat_Float32(depth_map);
        cv::Mat depth_cv_normalized, depth_clipped;
        // clip depth
        cv::threshold(depth_cv, depth_clipped, 10, 10, cv::THRESH_TRUNC);
        cv::threshold(depth_clipped, depth_clipped, 0, 0, cv::THRESH_TOZERO);

        // cv::normalize(depth_clipped, depth_cv_normalized, 0, 255, cv::NORM_MINMAX);
        // depth_cv_normalized.convertTo(depth_cv_normalized, CV_8UC1);
        // cv::imwrite(result_depth_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".png"), depth_cv_normalized);
    
        depth_clipped *= 6553.5;
        depth_clipped.convertTo(depth_clipped, CV_16U);
        cv::imwrite(result_depth_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".png"), depth_clipped);
    }
}

cv::Mat GaussianMapper::renderFromPose(
    const Sophus::SE3f &Tcw,
    const int width,
    const int height,
    const bool main_vision)
{
    if (!initial_mapped_ || getIteration() <= 0)
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>();
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    pkf->setPose(
        Tcw.unit_quaternion().cast<double>(),
        Tcw.translation().cast<double>());
    try {
        // Camera
        Camera& camera = scene_->cameras_.at(viewer_camera_id_);
        pkf->setCameraParams(camera);
        // Transformations
        pkf->computeTransformTensors();
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::renderFromPose]KeyFrame Camera not found!");
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> render_pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        // Render
        render_pkg = GaussianRenderer::render(
            pkf,
            height,
            width,
            gaussians_,
            pipe_params_,
            background_,
            override_color_
        );
    }

    // Result
    torch::Tensor masked_image;
    if (main_vision) {
        masked_image = std::get<0>(render_pkg) * viewer_main_undistort_mask_[pkf->camera_id_];
        masked_image = masked_image.index_select(0, torch::tensor({2, 1, 0}).to(device_type_)); // bgr rgb convert
        // std::cout << masked_image.size(0) << " " << masked_image.size(1) << " " << masked_image.size(2) << std::endl;
    }
    else
        masked_image = std::get<0>(render_pkg) * viewer_sub_undistort_mask_[pkf->camera_id_];
    return tensor_utils::torchTensor2CvMat_Float32(masked_image);
}

std::tuple<torch::Tensor, torch::Tensor> GaussianMapper::renderDepthFromPose(
    const Sophus::SE3d &Tcw,
    const int width,
    const int height,
    const bool main_vision)
{
    torch::NoGradGuard no_grad;

    // if (!initial_mapped_ || getIteration() <= 0)
    //     return cv::Mat(height, width, CV_32FC1, 0.0f);

    std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>();
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    pkf->setPose(
        Tcw.unit_quaternion(),
        Tcw.translation());

    try {
        // Camera
        Camera& camera = scene_->cameras_.at(0);
        pkf->setCameraParams(camera);
        // Transformations
        pkf->computeTransformTensors();
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::renderFromPose]KeyFrame Camera not found!");
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> render_pkg;
    {
        // std::unique_lock<std::mutex> lock_render(mutex_render_);
        // Render
        render_pkg = GaussianRenderer::render(
            pkf,
            height,
            width,
            gaussians_,
            pipe_params_,
            background_,
            override_color_
        );
    }

    // Result
    // torch::Tensor masked_image;
    // if (main_vision) {
    //     masked_image = std::get<0>(render_pkg) * viewer_main_undistort_mask_[pkf->camera_id_];
    //     masked_image = masked_image.index_select(0, torch::tensor({2, 1, 0}).to(device_type_)); // bgr rgb convert
    //     // std::cout << masked_image.size(0) << " " << masked_image.size(1) << " " << masked_image.size(2) << std::endl;
    // }
    // else
    //     masked_image = std::get<0>(render_pkg) * viewer_sub_undistort_mask_[pkf->camera_id_];
    
    torch::Tensor masked_depth, masked_alpha;
    // median depth
    // masked_depth = std::get<9>(render_pkg) * undistort_mask_[0];
    // masked_depth = std::get<9>(render_pkg);
    // expected depth
    masked_depth = std::get<7>(render_pkg) * undistort_mask_[0];
    // masked_depth = std::get<7>(render_pkg);

    masked_alpha = std::get<4>(render_pkg) * undistort_mask_[0];
    // masked_alpha = std::get<4>(render_pkg);

    std::tuple<torch::Tensor, torch::Tensor> out_info = std::make_tuple(masked_depth, masked_alpha);

    return out_info;
    // return tensor_utils::torchTensor2CvMat_Float32(masked_depth);
}

cv::Mat GaussianMapper::rendercvMatDepthFromPose(
    const Eigen::Matrix4f &Tcw_)
{
    torch::NoGradGuard no_grad;
    const int width = image_width;
    const int height = image_height;

    Sophus::SE3d Tcw(Tcw_.inverse().matrix().cast<double>());

    std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>();
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    pkf->setPose(
        Tcw.unit_quaternion(),
        Tcw.translation());

    try {
        // Camera
        Camera& camera = scene_->cameras_.at(0);
        pkf->setCameraParams(camera);
        // Transformations
        pkf->computeTransformTensors();
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::renderFromPose]KeyFrame Camera not found!");
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> render_pkg;
    {
        // std::unique_lock<std::mutex> lock_render(mutex_render_);
        // Render
        render_pkg = GaussianRenderer::render(
            pkf,
            height,
            width,
            gaussians_,
            pipe_params_,
            background_,
            override_color_
        );
    }

    
    torch::Tensor masked_depth;
    // median depth
    // masked_depth = std::get<9>(render_pkg) * undistort_mask_[pkf->camera_id_];
    // masked_depth = std::get<9>(render_pkg);
    // expected depth
    masked_depth = std::get<7>(render_pkg);

    // return masked_depth;
    return tensor_utils::torchTensor2CvMat_Float32(masked_depth);
}

void GaussianMapper::updateGSKeyFramesFromDSO()
{
    // -- DSO to 2DGS -- //
    // update sparse depth and keyframe poses
    boost::unique_lock<boost::mutex> frozenKFLock(pSLAM_->frozenMapMutex);
    
    kfid_in_dso_active_window.clear();

    for (dso::FrozenFrameHessian* kfs : pSLAM_->frameHessiansFrozen)
    {
        Sophus::SE3d c2w = kfs->camToWorld;
        auto gsKeyframe = scene_->getKeyframe(kfs->incomingID);

        if (gsKeyframe != nullptr) {
            // update pose
            gsKeyframe->setPose(
                c2w.unit_quaternion(),
                c2w.translation()
            );

            // update sparse depth
            gsKeyframe->sparse_depth_ =
                tensor_utils::cvMat2TorchTensor_Float32(kfs->kfSparseDepth, device_type_);

            kfid_in_dso_active_window.push_back(kfs->incomingID);
        }
    }
}

void GaussianMapper::updateKeyFramesFromGS()
{
    // -- 2DGS to DSO -- //
    // frameHessians lock
	boost::unique_lock<boost::mutex> lock(pSLAM_->mapMutex);

	std::vector<Sophus::SE3d> kf_poses;
	// get keyframe poses
	// for(dso::FrameHessian* kfs : pSLAM_->frameHessians)
	// {
	// 	Sophus::SE3d c2w = kfs->shell->camToWorld;
	// 	kf_poses.push_back(c2w);
	// }

    for (int i=0; i < pSLAM_->frameHessians.size()-2; i++)
	{
        auto kf = pSLAM_->frameHessians[i];
        Sophus::SE3d c2w = kf->shell->camToWorld;
        kf_poses.push_back(c2w);
    }

	// render depths (with keyframe poses)
	std::vector<std::tuple<torch::Tensor, torch::Tensor>> depth_maps = renderKFDepths(kf_poses);
	// get inverse depth points from depths & update current keyframe's idepth values
    for (int i=0; i < pSLAM_->frameHessians.size()-2; i++)
	{
        auto kf = pSLAM_->frameHessians[i];

        // std::cout << kf->pointHessians.size() << std::endl;

        // if (kf->pointHessians.empty())
        //     continue;

        torch::Tensor& renderedDepth = std::get<0>(depth_maps[i]);
        torch::Tensor& renderedAlpha = std::get<1>(depth_maps[i]);

        // int num_pp = 0;

		for (size_t j = 0; j < kf->pointHessians.size(); ++j) {
            dso::PointHessian* ph = kf->pointHessians[j];
            float stored_depth = 1.0f / ph->idepth_scaled;
            // float rendered_depth = renderedDepth.at<float>(ph->v, ph->u);
            // std::cout << renderedDepth.size(0) << " " << renderedDepth.size(1) << " " << renderedDepth.size(2) << std::endl;
            float rendered_depth = renderedDepth.index({0, int(ph->v), int(ph->u)}).item<float>();
            float rendered_alpha = renderedAlpha.index({0, int(ph->v), int(ph->u)}).item<float>();

            // if (false)
            if (std::abs(stored_depth-rendered_depth) < 0.1 and rendered_alpha > 0.9 and rendered_depth < 10.0 and rendered_depth > 0.01)
            {
                // std::cout << rendered_alpha << std::endl;
                // std::cout << std::abs(stored_depth-rendered_depth) << std::endl;
                float lambda = 0.5;
                float new_idepth = 1.0f / ((((1. - lambda) * stored_depth) + (lambda * rendered_depth)));
                ph->idepth_scaled = new_idepth;
                ph->idepth = new_idepth;
                // num_pp++;
            }
            // else {
            //     std::cout << rendered_alpha << std::endl;
            // }
        }

        // std::cout << float(num_pp) / kf->pointHessians.size() << std::endl;

        // for (size_t j = 0; j < kf->immaturePoints.size(); ++j) {
        //     dso::ImmaturePoint* ih = kf->immaturePoints[j];

        //     float rendered_alpha = renderedAlpha.index({0, int(ih->v), int(ih->u)}).item<float>();
        //     float rendered_depth = renderedDepth.index({0, int(ih->v), int(ih->u)}).item<float>();

        //     // std::cout << 1/ih->idepth_min << "/" << 1/ih->idepth_max << "/" << rendered_depth << std::endl;
        //     float depth_weight = 0.2;
            

        //     // is rendered depth valid?
        //     // if (rendered_alpha > 0.99 && rendered_depth > 0.1 && ih->quality > 5.) {
        //     if (rendered_alpha > 0.95 && rendered_depth > 0.01 && rendered_depth < 5.0) {
                
        //         float rendered_idepth = 1.0 / rendered_depth;

        //         ih->idepth_min = 1.0 / (rendered_depth + 0.1);
        //         ih->idepth_max = 1.0 / (rendered_depth - 0.1);

        //         // if (rendered_idepth > ih->idepth_min && rendered_idepth < ih->idepth_max) {
        //         //     // float dso_depth_min = 1.0 / ih->idepth_min;
        //         //     // float dso_depth_max = 1.0 / ih->idepth_max;

        //         //     // float new_idepth_min = 1.0 / (dso_depth_min + (depth_weight * (rendered_depth - dso_depth_min)));
        //         //     // float new_idepth_max = 1.0 / (dso_depth_max + (depth_weight * (dso_depth_max - rendered_depth)));

        //         //     // ih->idepth_min = new_idepth_min;
        //         //     // ih->idepth_max = new_idepth_max;
        //         // } 
        //         // else {
        //         //     // std::abs(dso_depth_max - dso_depth_min)
        //         //     ih->idepth_min = 1.0 / (rendered_depth + 0.1);
        //         //     ih->idepth_max = 1.0 / (rendered_depth - 0.1);
        //         // }
        //     }
        // }
	}
}

void GaussianMapper::renderAndRecordKeyframe(
    std::shared_ptr<GaussianKeyframe> pkf,
    float &dssim,
    float &psnr,
    float &psnr_gs,
    double &render_time,
    std::filesystem::path result_img_dir,
    std::filesystem::path result_gt_dir,
    std::filesystem::path result_loss_dir,
    std::filesystem::path result_depth_dir,
    std::string name_suffix)
{
    auto start_timing = std::chrono::steady_clock::now();
    auto render_pkg = GaussianRenderer::render(
        pkf,
        pkf->image_height_,
        pkf->image_width_,
        gaussians_,
        pipe_params_,
        background_,
        override_color_
    );
    auto rendered_image = std::get<0>(render_pkg);
    // expected depth
    auto surf_depth = std::get<7>(render_pkg);
    // median depth
    // auto surf_depth = std::get<9>(render_pkg);
    torch::Tensor masked_image = rendered_image * undistort_mask_[pkf->camera_id_];
    // torch::Tensor masked_dist = surf_depth * undistort_mask_[pkf->camera_id_];
    torch::Tensor masked_dist = surf_depth;
    torch::cuda::synchronize();
    auto end_timing = std::chrono::steady_clock::now();
    auto render_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_timing - start_timing).count();
    render_time = 1e-6 * render_time_ns;
    auto gt_image = pkf->original_image_;

    dssim = loss_utils::ssim(masked_image, gt_image, device_type_).item().toFloat();
    psnr = loss_utils::psnr(masked_image, gt_image).item().toFloat();
    // std::cout << masked_image.sizes() << std::endl;
    // std::cout << gt_image.sizes() << std::endl;
    psnr_gs = loss_utils::psnr_gaussian_splatting(masked_image, gt_image).item().toFloat();

    recordKeyframeRendered(masked_image, gt_image, masked_dist, pkf->fid_, result_img_dir, result_gt_dir, result_loss_dir, result_depth_dir, name_suffix);    
}

void GaussianMapper::renderAndRecordAllKeyframes(
    std::string name_suffix)
{
    std::filesystem::path result_dir = result_dir_ / (std::to_string(getIteration()) + name_suffix);
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path image_dir = result_dir / "image";
    if (record_rendered_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_dir);

    std::filesystem::path image_gt_dir = result_dir / "image_gt";
    if (record_ground_truth_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_gt_dir);

    std::filesystem::path image_loss_dir = result_dir / "image_loss";
    if (record_loss_image_) {
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_loss_dir);
    }

    std::filesystem::path depth_dir = result_dir / "depth";
    if (record_depth_image_) {
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(depth_dir);
    }

    std::filesystem::path render_time_path = result_dir / "render_time.txt";
    std::ofstream out_time(render_time_path);
    out_time << "##[Gaussian Mapper]Render time statistics: keyframe id, time(milliseconds)" << std::endl;

    std::filesystem::path dssim_path = result_dir / "dssim.txt";
    std::ofstream out_dssim(dssim_path);
    out_dssim << "##[Gaussian Mapper]keyframe id, dssim" << std::endl;

    std::filesystem::path psnr_path = result_dir / "psnr.txt";
    std::ofstream out_psnr(psnr_path);
    out_psnr << "##[Gaussian Mapper]keyframe id, psnr" << std::endl;

    std::filesystem::path psnr_gs_path = result_dir / "psnr_gaussian_splatting.txt";
    std::ofstream out_psnr_gs(psnr_gs_path);
    out_psnr_gs << "##[Gaussian Mapper]keyframe id, psnr_gaussian_splatting" << std::endl;

    std::size_t nkfs = scene_->keyframes().size();
    auto kfit = scene_->keyframes().begin();
    float dssim, psnr, psnr_gs;
    double render_time;
    for (std::size_t i = 0; i < nkfs; ++i) {
        renderAndRecordKeyframe((*kfit).second, dssim, psnr, psnr_gs, render_time, image_dir, image_gt_dir, image_loss_dir, depth_dir);
        out_time << (*kfit).first << " " << std::fixed << std::setprecision(8) << render_time << std::endl;

        out_dssim   << (*kfit).first << " " << std::fixed << std::setprecision(10) << dssim   << std::endl;
        out_psnr    << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr    << std::endl;
        out_psnr_gs << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr_gs << std::endl;

        ++kfit;
    }
}

void GaussianMapper::savePly(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    keyframesToJson(result_dir);
    saveModelParams(result_dir);

    gaussians_->savePly(result_dir / "point_cloud.ply");
    gaussians_->saveSparsePointsPly(result_dir / "input.ply");
}

void GaussianMapper::keyframesToJson(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path result_path = result_dir / "cameras.json";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json file at " + result_path.string());

    Json::Value json_root;
    Json::StreamWriterBuilder builder;
    const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

    int i = 0;
    for (const auto& kfit : scene_->keyframes()) {
        const auto pkf = kfit.second;
        Eigen::Matrix4f Rt;
        Rt.setZero();
        Eigen::Matrix3f R = pkf->R_quaternion_.toRotationMatrix().cast<float>();
        Rt.topLeftCorner<3, 3>() = R;
        Eigen::Vector3f t = pkf->t_.cast<float>();
        Rt.topRightCorner<3, 1>() = t;
        Rt(3, 3) = 1.0f;

        Eigen::Matrix4f Twc = Rt.inverse();
        Eigen::Vector3f pos = Twc.block<3, 1>(0, 3);
        Eigen::Matrix3f rot = Twc.block<3, 3>(0, 0);

        Json::Value json_kf;
        json_kf["id"] = static_cast<Json::Value::UInt64>(pkf->fid_);
        json_kf["img_name"] = pkf->img_filename_; //(std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_));
        json_kf["width"] = pkf->image_width_;
        json_kf["height"] = pkf->image_height_;

        json_kf["position"][0] = pos.x();
        json_kf["position"][1] = pos.y();
        json_kf["position"][2] = pos.z();

        json_kf["rotation"][0][0] = rot(0, 0);
        json_kf["rotation"][0][1] = rot(0, 1);
        json_kf["rotation"][0][2] = rot(0, 2);
        json_kf["rotation"][1][0] = rot(1, 0);
        json_kf["rotation"][1][1] = rot(1, 1);
        json_kf["rotation"][1][2] = rot(1, 2);
        json_kf["rotation"][2][0] = rot(2, 0);
        json_kf["rotation"][2][1] = rot(2, 1);
        json_kf["rotation"][2][2] = rot(2, 2);

        json_kf["fy"] = graphics_utils::fov2focal(pkf->FoVy_, pkf->image_height_);
        json_kf["fx"] = graphics_utils::fov2focal(pkf->FoVx_, pkf->image_width_);

        json_root[i] = Json::Value(json_kf);
        ++i;
    }

    writer->write(json_root, &out_stream);
}

void GaussianMapper::saveModelParams(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / "cfg_args";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open file at " + result_path.string());

    out_stream << "Namespace("
               << "eval=" << (model_params_.eval_ ? "True" : "False") << ", "
               << "images=" << "\'" << model_params_.images_ << "\', "
               << "model_path=" << "\'" << model_params_.model_path_.string() << "\', "
               << "resolution=" << model_params_.resolution_ << ", "
               << "sh_degree=" << model_params_.sh_degree_ << ", "
               << "source_path=" << "\'" << model_params_.source_path_.string() << "\', "
               << "white_background=" << (model_params_.white_background_ ? "True" : "False") << ", "
               << ")";

    out_stream.close();
}

void GaussianMapper::writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / ("keyframe_used_times" + name_suffix + ".txt");
    std::ofstream out_stream;
    out_stream.open(result_path, std::ios::app);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json at " + result_path.string());

    out_stream << "##[Gaussian Mapper]Iteration " << getIteration() << " keyframe id, used times, remaining times:\n";
    for (const auto& used_times_it : kfs_used_times_)
        out_stream << used_times_it.first << " "
                   << used_times_it.second << " "
                   << scene_->keyframes().at(used_times_it.first)->remaining_times_of_use_
                   << "\n";
    out_stream << "##=========================================" <<std::endl;

    out_stream.close();
}

int GaussianMapper::getIteration()
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    return iteration_;
}
void GaussianMapper::increaseIteration(const int inc)
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    iteration_ += inc;
}

float GaussianMapper::positionLearningRateInit()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.position_lr_init_;
}
float GaussianMapper::featureLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.feature_lr_;
}
float GaussianMapper::opacityLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_lr_;
}
float GaussianMapper::scalingLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.scaling_lr_;
}
float GaussianMapper::rotationLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.rotation_lr_;
}
float GaussianMapper::percentDense()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.percent_dense_;
}
float GaussianMapper::lambdaDssim()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.lambda_dssim_;
}
int GaussianMapper::opacityResetInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_reset_interval_;
}
float GaussianMapper::densifyGradThreshold()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densify_grad_threshold_;
}
int GaussianMapper::densifyInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densification_interval_;
}
int GaussianMapper::newKeyframeTimesOfUse()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return new_keyframe_times_of_use_;
}
int GaussianMapper::stableNumIterExistence()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return stable_num_iter_existence_;
}
bool GaussianMapper::isKeepingTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return keep_training_;
}
bool GaussianMapper::isdoingGausPyramidTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return do_gaus_pyramid_training_;
}
bool GaussianMapper::isdoingInactiveGeoDensify()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return inactive_geo_densify_;
}

void GaussianMapper::setPositionLearningRateInit(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.position_lr_init_ = lr;
}
void GaussianMapper::setFeatureLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.feature_lr_ = lr;
}
void GaussianMapper::setOpacityLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_lr_ = lr;
}
void GaussianMapper::setScalingLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.scaling_lr_ = lr;
}
void GaussianMapper::setRotationLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.rotation_lr_ = lr;
}
void GaussianMapper::setPercentDense(const float percent_dense)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.percent_dense_ = percent_dense;
    gaussians_->setPercentDense(percent_dense);
}
void GaussianMapper::setLambdaDssim(const float lambda_dssim)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.lambda_dssim_ = lambda_dssim;
}
void GaussianMapper::setOpacityResetInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_reset_interval_ = interval;
}
void GaussianMapper::setDensifyGradThreshold(const float th)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densify_grad_threshold_ = th;
}
void GaussianMapper::setDensifyInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densification_interval_ = interval;
}
void GaussianMapper::setNewKeyframeTimesOfUse(const int times)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    new_keyframe_times_of_use_ = times;
}
void GaussianMapper::setStableNumIterExistence(const int niter)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    stable_num_iter_existence_ = niter;
}
void GaussianMapper::setKeepTraining(const bool keep)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    keep_training_ = keep;
}
void GaussianMapper::setDoGausPyramidTraining(const bool gaus_pyramid)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    do_gaus_pyramid_training_ = gaus_pyramid;
}
void GaussianMapper::setDoInactiveGeoDensify(const bool inactive_geo_densify)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    inactive_geo_densify_ = inactive_geo_densify;
}

VariableParameters GaussianMapper::getVaribleParameters()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    VariableParameters params;
    params.position_lr_init = opt_params_.position_lr_init_;
    params.feature_lr = opt_params_.feature_lr_;
    params.opacity_lr = opt_params_.opacity_lr_;
    params.scaling_lr = opt_params_.scaling_lr_;
    params.rotation_lr = opt_params_.rotation_lr_;
    params.percent_dense = opt_params_.percent_dense_;
    params.lambda_dssim = opt_params_.lambda_dssim_;
    params.opacity_reset_interval = opt_params_.opacity_reset_interval_;
    params.densify_grad_th = opt_params_.densify_grad_threshold_;
    params.densify_interval = opt_params_.densification_interval_;
    params.new_kf_times_of_use = new_keyframe_times_of_use_;
    params.stable_num_iter_existence = stable_num_iter_existence_;
    params.keep_training = keep_training_;
    params.do_gaus_pyramid_training = do_gaus_pyramid_training_;
    params.do_inactive_geo_densify = inactive_geo_densify_;
    return params;
}

void GaussianMapper::setVaribleParameters(const VariableParameters &params)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.position_lr_init_ = params.position_lr_init;
    opt_params_.feature_lr_ = params.feature_lr;
    opt_params_.opacity_lr_ = params.opacity_lr;
    opt_params_.scaling_lr_ = params.scaling_lr;
    opt_params_.rotation_lr_ = params.rotation_lr;
    opt_params_.percent_dense_ = params.percent_dense;
    gaussians_->setPercentDense(params.percent_dense);
    opt_params_.lambda_dssim_ = params.lambda_dssim;
    opt_params_.opacity_reset_interval_ = params.opacity_reset_interval;
    opt_params_.densify_grad_threshold_ = params.densify_grad_th;
    opt_params_.densification_interval_ = params.densify_interval;
    new_keyframe_times_of_use_ = params.new_kf_times_of_use;
    stable_num_iter_existence_ = params.stable_num_iter_existence;
    keep_training_ = params.keep_training;
    do_gaus_pyramid_training_ = params.do_gaus_pyramid_training;
    inactive_geo_densify_ = params.do_inactive_geo_densify;
}

void GaussianMapper::loadPly(std::filesystem::path ply_path, std::filesystem::path camera_path)
{
    this->gaussians_->loadPly(ply_path);

    // Camera
    if (!camera_path.empty() && std::filesystem::exists(camera_path)) {
        cv::FileStorage camera_file(camera_path.string().c_str(), cv::FileStorage::READ);
        if(!camera_file.isOpened())
            throw std::runtime_error("[Gaussian Mapper]Failed to open settings file at: " + camera_path.string());

        Camera camera;
        camera.camera_id_ = 0;
        camera.width_ = camera_file["Camera.w"].operator int();
        camera.height_ = camera_file["Camera.h"].operator int();

        std::string camera_type = camera_file["Camera.type"].string();
        if (camera_type == "Pinhole") {
            camera.setModelId(Camera::CameraModelType::PINHOLE);

            float fx = camera_file["Camera.fx"].operator float();
            float fy = camera_file["Camera.fy"].operator float();
            float cx = camera_file["Camera.cx"].operator float();
            float cy = camera_file["Camera.cy"].operator float();

            float k1 = camera_file["Camera.k1"].operator float();
            float k2 = camera_file["Camera.k2"].operator float();
            float p1 = camera_file["Camera.p1"].operator float();
            float p2 = camera_file["Camera.p2"].operator float();
            float k3 = camera_file["Camera.k3"].operator float();

            cv::Mat K = (
                cv::Mat_<float>(3, 3)
                    << fx, 0.f, cx,
                        0.f, fy, cy,
                        0.f, 0.f, 1.f
            );

            camera.params_[0] = fx;
            camera.params_[1] = fy;
            camera.params_[2] = cx;
            camera.params_[3] = cy;

            std::vector<float> dist_coeff = {k1, k2, p1, p2, k3};
            camera.dist_coeff_ = cv::Mat(5, 1, CV_32F, dist_coeff.data());
            camera.initUndistortRectifyMapAndMask(K, cv::Size(camera.width_, camera.height_), K, false);

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                       cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);

        }
        else {
            throw std::runtime_error("[Gaussian Mapper]Unsupported camera model: " + camera_path.string());
        }

        if (!viewer_camera_id_set_) {
            viewer_camera_id_ = camera.camera_id_;
            viewer_camera_id_set_ = true;
        }
        this->scene_->addCamera(camera);
    }

    // Ready
    this->initial_mapped_ = true;
    increaseIteration();
}