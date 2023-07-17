%{
  CSI Feature Extraction for Wi-Fi sensing.  
  - Input: csi source data.
  - Output: extracted features.

  To use this script, you need to:
  1. Make sure the csi data have been saved as .mat files.
  2. Check the .mat file to make sure they are in the correct form. 
  3. Set the parameters.
  
  Note that in this algorithm, the csi data should be a 4-D tensor with the size of [T S A L]:
  - T indicates the number of csi packets;
  - S indicates the number of subcarriers;
  - A indicates the number of antennas (i.e., the STS number in a MIMO system);
  - L indicates the number of HT-LTFs in a single PPDU;
  Say if we collect a 10 seconds csi data steam at 1 kHz sample rate (T = 10 * 1000), from a 3-antenna AP (A = 3),  with 802.11n standard (S = 57 subcarrier), without any extra spatial sounding (L = 1), the data size should be [10000 57 3 1].
%}

clear all;
addpath(genpath(pwd));

%% 0. Set parameters.
% Path of the raw CSI data.
src_file_name = './data/csi_src_test.mat';
% Path of the saved CSI features.
dst_file_name = './data/csi_feature_test.mat';

% Speed of light.
global c;
c = physconst('LightSpeed');
% Bandwidth.
global bw;
bw = 20e6;
% Subcarrier frequency.
global subcarrier_freq;
subcarrier_freq = linspace(5.8153e9, 5.8347e9, 57);
% Subcarrier wavelength.
global subcarrier_lambda;
subcarrier_lambda = c ./ subcarrier_freq;

% Antenna arrangement.
antenna_loc = [0, 0, 0; 0.0514665, 0, 0; 0, 0.0514665, 0]';
% Set the linear range of the CSI phase, which varies with NIC types.
linear_interval = (20:38)';

%% 1. Read the csi data for calibration and sanitization.
% Load the raw CSI data.
csi_src = load(src_file_name).csi;      % Raw CSI.

%% 2. Extract wireless features.
% Test example 1: angle/direction estimation with imperfect CSI.
[packet_num, subcarrier_num, antenna_num, ~] = size(csi_src);
aoa_mat = naive_aoa(csi_src, antenna_loc, zeros(3, 1));
aoa_gt = [0; 0; 1];
error = mean(acos(aoa_gt' * aoa_mat));
disp("Angle estimation error: " + num2str(error));

% Test example 2: distance estimation with CSI.
tof_mat = naive_tof(csi_src);
est_dist = mean(tof_mat * c, 'all');
disp("The ground truth distance is: 10 m");
disp("The estimated distance is: " + num2str(est_dist) + " m");

% Test example 3: 
sample_rate = 50;
visable = true;
stft_mat = naive_spectrum(csi_src, sample_rate, visable);

%% 3. Save the extracted feature as needed.
save(dst_file_name, 'aoa_mat', 'tof_mat', 'stft_mat');

%% Please refer to our website page or our tutorial paper on arXiv for more detailed information. 