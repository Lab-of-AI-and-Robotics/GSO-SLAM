/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023,
 * as part of Photo-SLAM.
 */

#include "include/gaussian_renderer.h"

torch::Tensor depths_to_points(std::shared_ptr<GaussianKeyframe> view, 
                                const torch::Tensor& depthmap) {

    auto c2w = view->world_view_transform_.transpose(0, 1).inverse();
    // for pyramid gs optimization
    // float W = (float)view->image_width_;
    // float H = (float)view->image_height_;
    float W = depthmap.sizes()[2];
    float H = depthmap.sizes()[1];

    // std::cout << depthmap.sizes() << std::endl;

    // NDC to pixel transform
    torch::Tensor ndc2pix = torch::tensor({
        {W / 2.0, 0.0,     0.0, W / 2.0},
        {0.0,     H / 2.0, 0.0, H / 2.0},
        {0.0,     0.0,     0.0, 1.0}}).to(torch::kCUDA).to(torch::kFloat).transpose(0, 1);

    auto projection_matrix = c2w.transpose(0, 1).matmul(view->full_proj_transform_);
    auto intrins = projection_matrix.matmul(ndc2pix).index({torch::indexing::Slice(0, 3), 
                                                            torch::indexing::Slice(0, 3)}).transpose(0, 1);

    auto grid_x = torch::arange(0, W, torch::kFloat).to(torch::kCUDA);
    auto grid_y = torch::arange(0, H, torch::kFloat).to(torch::kCUDA);
    auto meshgrid = torch::meshgrid({grid_x, grid_y}, /*indexing=*/"xy");

    auto points = torch::stack({meshgrid[0], meshgrid[1], torch::ones_like(meshgrid[0])}, -1)
                    .reshape({-1, 3});
    auto rays_d = points.matmul(intrins.inverse().transpose(0, 1))
                         .matmul(c2w.index({torch::indexing::Slice(0, 3), 
                                            torch::indexing::Slice(0, 3)}).transpose(0, 1));
    auto rays_o = c2w.index({torch::indexing::Slice(0, 3), 3});
    auto transformed_points = depthmap.reshape({-1, 1}) * rays_d + rays_o;

    return transformed_points;
}

torch::Tensor depth_to_normal(std::shared_ptr<GaussianKeyframe> view, 
                               const torch::Tensor& depthmap) {

    auto points = depths_to_points(view, depthmap)
                    .reshape({depthmap.size(1), depthmap.size(2), 3});
    auto output = torch::zeros_like(points);


    auto dx = points.index({torch::indexing::Slice(2, torch::indexing::None), 
                            torch::indexing::Slice(1, -1)}) - 
              points.index({torch::indexing::Slice(torch::indexing::None, -2), 
                            torch::indexing::Slice(1, -1)});


    auto dy = points.index({torch::indexing::Slice(1, -1), 
                            torch::indexing::Slice(2, torch::indexing::None)}) - 
              points.index({torch::indexing::Slice(1, -1), 
                            torch::indexing::Slice(torch::indexing::None, -2)});

    auto normal_map = torch::nn::functional::normalize(torch::cross(dx, dy, -1), 
                    torch::nn::functional::NormalizeFuncOptions().dim(-1));

    output.index_put_({torch::indexing::Slice(1, -1), torch::indexing::Slice(1, -1)}, normal_map);

    return output;
}

/**
 * @brief 
 * 
 * @return std::tuple<render, viewspace_points, visibility_filter, radii>, which are all `torch::Tensor`
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianRenderer::render(
    std::shared_ptr<GaussianKeyframe> viewpoint_camera,
    int image_height,
    int image_width,
    std::shared_ptr<GaussianModel> pc,
    GaussianPipelineParams& pipe,
    torch::Tensor& bg_color,
    torch::Tensor& override_color,
    float scaling_modifier,
    bool use_override_color)
{
    /* Render the scene. 

       Background tensor (bg_color) must be on GPU!
     */

    // Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    auto screenspace_points = torch::zeros_like(pc->getXYZ(),
        torch::TensorOptions().dtype(pc->getXYZ().dtype()).requires_grad(true).device(torch::kCUDA));
    try {
        screenspace_points.retain_grad();
    }
    catch (const std::exception& e) {
        ; // pass
    }

    // Set up rasterization configuration
    float tanfovx = std::tan(viewpoint_camera->FoVx_ * 0.5f);
    float tanfovy = std::tan(viewpoint_camera->FoVy_ * 0.5f);

    GaussianRasterizationSettings raster_settings(
        image_height,
        image_width,
        tanfovx,
        tanfovy,
        bg_color,
        scaling_modifier,
        viewpoint_camera->world_view_transform_,
        viewpoint_camera->full_proj_transform_,
        pc->active_sh_degree_,
        viewpoint_camera->camera_center_,
        false
    );

    GaussianRasterizer rasterizer(raster_settings);

    auto means3D = pc->getXYZ();
    auto means2D = screenspace_points;
    auto opacity = pc->getOpacityActivation();

    /* If precomputed 3d covariance is provided, use it. If not, then it will be computed from
       scaling / rotation by the rasterizer. 
     */
    bool has_scales = false,
         has_rotations = false,
         has_cov3D_precomp = false;
    torch::Tensor scales,
                  rotations,
                  cov3D_precomp;
    if (pipe.compute_cov3D_) {
        cov3D_precomp = pc->getCovarianceActivation();
        has_cov3D_precomp = true;
    }
    else {
        scales = pc->getScalingActivation();
        rotations = pc->getRotationActivation();
        has_scales = true;
        has_rotations = true;
    }

    /* If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
       from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
     */
    bool has_shs = false,
         has_color_precomp = false;
    torch::Tensor shs,
                  colors_precomp;
    if (use_override_color) {
        colors_precomp = override_color;
        has_color_precomp = true;
    }
    else {
        if (pipe.convert_SHs_) {
            int max_sh_degree = pc->max_sh_degree_ + 1;
            torch::Tensor shs_view = pc->getFeatures().transpose(1, 2).view({-1, 3, max_sh_degree * max_sh_degree});
            torch::Tensor dir_pp = (pc->getXYZ() - viewpoint_camera->camera_center_.repeat({pc->getFeatures().size(0), 1}));
            auto dir_pp_normalized = dir_pp / torch::frobenius_norm(dir_pp, /*dim=*/{1}, /*keepdim=*/true);
            auto sh2rgb = sh_utils::eval_sh(pc->active_sh_degree_, shs_view, dir_pp_normalized);
            colors_precomp = torch::clamp_min(sh2rgb + 0.5, 0.0);
            has_color_precomp = true;
        }
        else {
            shs = pc->getFeatures();
            has_shs = true;
        }
    }

    // Rasterize visible Gaussians to image, obtain their radii (on screen). 
    auto rasterizer_result = rasterizer.forward(
        means3D,
        means2D,
        opacity,
        has_shs,
        has_color_precomp,
        has_scales,
        has_rotations,
        has_cov3D_precomp,
        shs,
        colors_precomp,
        scales,
        rotations,
        cov3D_precomp
    );
    auto rendered_image = std::get<0>(rasterizer_result);
    auto radii = std::get<1>(rasterizer_result);
    auto allmap = std::get<2>(rasterizer_result);

    // additional regularizations
    auto render_alpha = allmap.index({torch::indexing::Slice(1, 2)});

    // normal map
    // transform normal from view space to world space
    auto render_normal = allmap.index({torch::indexing::Slice(2, 5)});
    render_normal = render_normal.permute({1, 2, 0});
    torch::Tensor rotation_matrix = viewpoint_camera->world_view_transform_.index({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}).transpose(0, 1);
    render_normal = torch::matmul(render_normal, rotation_matrix);
    render_normal = render_normal.permute({2, 0, 1});

    // median depth
    auto render_depth_median = allmap.index({torch::indexing::Slice(5, 6)});
    render_depth_median = torch::nan_to_num(render_depth_median, 0, 0);

    // expected depth map
    auto render_depth_expected = allmap.index({torch::indexing::Slice(0, 1)});
    render_depth_expected = (render_depth_expected / render_alpha);
    render_depth_expected = torch::nan_to_num(render_depth_expected, 0, 0);

    // depth distortion map
    auto render_dist = allmap.index({torch::indexing::Slice(6, 7)});

    // psedo surface attributes
    // surf depth is either median or expected by setting depth_ratio to 1 or 0
    // for bounded scene, use median depth, i.e., depth_ratio = 1; 
    // for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    auto surf_depth = render_depth_expected * (1-pipe.depth_ratio_) + (pipe.depth_ratio_) * render_depth_median;
    
    // assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    auto surf_normal = depth_to_normal(viewpoint_camera, surf_depth);
    surf_normal = surf_normal.permute({2,0,1});
    // remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach();

    /* Those Gaussians that were frustum culled or had a radius of 0 were not visible.
       They will be excluded from value updates used in the splitting criteria.
     */
    return std::make_tuple(
        rendered_image,     /*render*/
        screenspace_points, /*viewspace_points*/
        radii > 0,          /*visibility_filter*/
        radii,               /*radii*/
        render_alpha,
        render_normal,
        render_dist,
        surf_depth,
        surf_normal,
        render_depth_median
    );
}
