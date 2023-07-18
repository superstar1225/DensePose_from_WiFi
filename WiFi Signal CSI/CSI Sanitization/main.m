%{
  CSI Sanitization Algorithm for Wi-Fi sensing.  
  - Input: csi data used for calibration, and csi data that need to be sanitized.
  - Output: sanitized csi data.

  To use this script, you need to:
  1. Make sure the csi data have been saved as .mat files.
  2. Check the .mat file to make sure they are in the correct form. 
  3. Set the parameters.
  
  Note that in this algorithm, the csi data should be a 4-D tensor with the size of [T S A L]:
  - T indicates the number of csi packets;
  - S indicates the number of subcarriers;
  - A indicates the number of antennas (i.e., the STS number in a MIMO system);
  - L indicates the number of HT-LTFs in a single PPDU;
  Say if we collect a 10 seconds csi data steam at 1 kHz sample rate (T = 10 * 1000), from a 3-antenna AP (A = 3),  with 802.11n standard (S = 57 subcarrier), without only one spatial stream (L = 1), the data size should be [10000 57 3 1].
%}

clear all;
addpath(genpath(pwd));

%% 0. Set parameters.
% Path of the calibration data;
calib_file_name = './data/csi_calib_test.mat';
% Path for storing the generated calibration templated.
calib_template_name = './data/calib_template_test.mat';
% Path of the raw CSI data.
src_file_name = './data/csi_src_test.mat';
% Path for storing the sanitized CSI data.
dst_file_name = './data/csi_dst_test.mat';

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
% Load the calibration data. 
csi_calib = load(calib_file_name).csi; % CSI for calibration.
% Load the raw CSI data.
csi_src = load(src_file_name).csi;      % Raw CSI.

%% 2. Choose different functions according to your task.
% Use cases:
% Make calibration template.
csi_calib_template = set_template(csi_calib, linear_interval, calib_template_name);
% Directly load the generated template.
csi_calib_template = load(calib_template_name).csi;
% Remove the nonlinear error.
csi_remove_nonlinear = nonlinear_calib(csi_src, csi_calib_template);
% Remove the STO (a.k.a SFO and PBD) by conjugate mulitplication.
csi_remove_sto = sto_calib_mul(csi_src);
% Remove the STO (a.k.a SFO and PBD) by conjugate division.
csi_remove_sto = sto_calib_div(csi_src);
% Estimate the CFO by frequency tracking.
est_cfo = cfo_calib(csi_src);
% Estimate the RCO.
est_rco = rco_calib(csi_calib);

%% 3. Save the sanitized data as needed.
csi = csi_remove_sto;
save(dst_file_name, 'csi');

%% 4. Perform various wireless sensing tasks.
% Test example 1: angle/direction estimation with imperfect CSI.
[packet_num, subcarrier_num, antenna_num, ~] = size(csi_src);
est_rco = rco_calib(csi_calib);
zero_rco = zeros(antenna_num, 1);
aoa_mat_error = naive_aoa(csi_src, antenna_loc, zero_rco);
aoa_mat = naive_aoa(csi_src, antenna_loc, est_rco);
aoa_gt = [0; 0; 1];
error_1 = mean(acos(aoa_gt' * aoa_mat_error));
error_2 = mean(acos(aoa_gt' * aoa_mat));
disp("Angle estimation error (in deg) without RCO removal: " + num2str(error_1));
disp("Angle estimation error (in deg) with RCO removal: " + num2str(error_2));

% Test example 2: intrusion detection with CSI.
csi_sto_calib = sto_calib_div(csi_src);
intrusion_flag_raw = naive_intrusion(csi_src, 3);
intrusion_flag_sanitized = naive_intrusion(csi_sto_calib, 3);
disp("Intrusion detection result without SFO/PDD removal: " + num2str(intrusion_flag_raw));
disp("Intrusion detection result with SFO/PDD removal: " + num2str(intrusion_flag_sanitized));

%% Please refer to our website page or our tutorial paper on arXiv for more detailed information. 