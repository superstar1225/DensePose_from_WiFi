function [est_cfo] = cfo_calib(csi_src)
    % cfo_calib
    % Input:
    %   - csi_src is the csi data with two HT-LTFs; [T S A L]
    % Output:
    %   - est_cfo is the estimated frequency offset; 

    delta_time = 4e-6;
    phase_1 = angle(csi_src(:, :, :, 1));
    phase_2 = angle(csi_src(:, :, :, 2));
    phase_diff = mean(phase_2 - phase_1, 3); % [T S A 1]
    est_cfo = mean(phase_diff ./ delta_time, 2);
end