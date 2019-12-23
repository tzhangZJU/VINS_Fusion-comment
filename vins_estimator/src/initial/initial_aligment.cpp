/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "initial_alignment.h"

void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)  //计算Bg、并更新图像帧对应的预积分
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++) //遍历所有图像帧及图像帧之间的预积分
    {
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);  //R_bi_bj = R_w_bi^T*R_w_bj  tzhang
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);  //旋转量相对于bg的雅克比J tzhang
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();  //残差值
        A += tmp_A.transpose() * tmp_A;  //J^T*J  构建normal equation tzhang
        b += tmp_A.transpose() * tmp_b;  //J^T*b
    }
    delta_bg = A.ldlt().solve(b);  //基于Cholesky的线性方程组求解陀螺仪偏置Bg变化量 tzhang
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;  //利用计算的Bg变化量，更新Bg

    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++) //由于Bg发生了变化，需要对图像帧对应的预积分进行再次传播计算
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);  //TODO(tzhang): Why：solveGyroscopeBias函数中全部是基于Bgs[0]进行的预积分 tzhang
        //而Estimator中的预积分并不是这样处理的，具体见pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
}


MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();  //使用前述优化得到的重力方向、G的模值，构建新的重力向量
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;  //此时限定重力模值，仅优化重力方向，因此重力相关的自由度为2

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for(int k = 0; k < 4; k++)  //进行4次迭代优化计算，迭代过程中仅更新重力向量
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9);  //公式（19）H（重力相关自由度变为2）
            tmp_A.setZero();
            VectorXd tmp_b(6);  //公式（18）Z（重力相关自由度变为2）
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;


            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;  //重力相关的雅克比发生变化
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;  //重力相关的雅克比发生变化
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);
            VectorXd dg = x.segment<2>(n_state - 3);
            g0 = (g0 + lxly * dg).normalized() * G.norm();  //更新重力向量
            //double s = x(n_state - 1);
    }   
    g = g0;  //四次迭代优化的重力向量
}

bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1;  //该优化过程中状态向量维度，对应公式（16），速度（3*图像帧数目）、重力（3）、尺度因子（1）

    MatrixXd A{n_state, n_state};  // Hessian矩阵
    A.setZero();
    VectorXd b{n_state};  //normal-equation方程右侧向量
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 10);  //公式（19）H
        tmp_A.setZero();
        VectorXd tmp_b(6);  //公式（18）Z
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();  //分别对应公式18、19的内容
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;  // 10x10
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;  //10x1

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();  //速度相关部分，k、k+1时刻速度，共6维
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();  //重力向量与尺度因子相关部分，共4维
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;  //同时乘以1000，提高数值稳定性
    b = b * 1000.0;
    x = A.ldlt().solve(b);  //求解normal-equation
    double s = x(n_state - 1) / 100.0;  //尺度因子s
    ROS_DEBUG("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);  //重力向量
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    if(fabs(g.norm() - G.norm()) > 0.5 || s < 0)  //如果重力向量与设定的G模值相差较大，或者尺度因子s为负数，认为失败
    {
        return false;
    }

    RefineGravity(all_image_frame, g, x);  //利用重力向量的模值信息，进一步优化；优化方法与前述基本一致，仅重力向量优化维度减小为2
    s = (x.tail<1>())(0) / 100.0;  //TODO(tzhang):尺度因子处理，除以了100,（PS：100在tmp_A.block<3, 1>(0, 9)有所体现，但具体为何如何处理还未知）
    (x.tail<1>())(0) = s;
    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0 )
        return false;   
    else
        return true;
}

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    solveGyroscopeBias(all_image_frame, Bgs);  //计算Bg、并更新图像帧对应的预积分 对应vins-mono公式（15）

    if(LinearAlignment(all_image_frame, g, x))  //imu与camera对准，及速度、尺度、重力向量计算，对应vins-mono公式（16~20）
        return true;
    else 
        return false;
}
