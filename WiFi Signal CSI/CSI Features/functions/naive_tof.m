function [tof_mat] = naive_tof(csi_data)
    % naive_tof
    % Input:
    %   - csi_data is the CSI used for ranging; [T S A L]
    % Output:
    %   - tof_mat is the rough time-of-flight estimation result; [T A]

    % The bandwidth parameter.
    global bw;
    [~, subcarrier_num, ~, ~] = size(csi_data);
    % Exponential powers of 2, based on the rounded up subcarrier number.
    ifft_point = power(2, ceil(log2(subcarrier_num)));
    % Get CIR from each packet and each antenna by ifft(CFR);
    cir_sequence = ifft(csi_data, ifft_point, 2); % [T ifft_point A L]
    cir_sequence = squeeze(mean(cir_sequence, 4)); % [T ifft_point A]
    % Only consider half of the ifft points.
    half_point = ifft_point / 2;
    half_sequence = cir_sequence(:, 1:half_point, :); % [T half_point A]
    % Find the peak of the CIR sequence.
    [~, peak_indices] = max(half_sequence, [], 2); % [T 1 A]
    peak_indices = squeeze(peak_indices); % [T A]
    % Calculate ToF for each packet and each antenna, based on the CIR peak.
    tof_mat = peak_indices .* subcarrier_num ./ (ifft_point .* bw); % [T A]
end