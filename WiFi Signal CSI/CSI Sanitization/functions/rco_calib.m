function [est_rco] = rco_calib(csi_calib)
    % rco_calib
    % Input:
    %   - csi_calib is the reference csi at given distance and angle; [T S A L]
    % Output:
    %   - est_rco is the estimated RCO; [A 1]

    antenna_num = size(csi_calib, 3);
    csi_phase = unwrap(angle(csi_calib), [], 1);    % [T S A L]
    avg_phase = zeros(antenna_num, 1);
    for a = 1:antenna_num
        avg_phase(a, 1) = mean(csi_phase(:, :, a, 1), 'all');
    end
    est_rco = avg_phase - avg_phase(1, 1);
end