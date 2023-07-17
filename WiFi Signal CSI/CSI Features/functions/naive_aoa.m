function [aoa_mat] = naive_aoa(csi_data, antenna_loc, est_rco)
    % naive_aoa
    % Input:
    %   - csi_data is the CSI used for angle estimation; [T S A L]
    %   - antenna_loc is the antenna location arrangement with the first antenna as a reference; [3 A]
    %   - est_rco is the estimated radio chain offset; [A 1]
    %     you may ignore the est_rco if you don't know what it is. RCO will be introduced in Sec. 5.
    % Output:
    %   - aoa_mat is the angle estimation result; [3 T]

    global subcarrier_lambda;
    csi_phase = unwrap(angle(csi_data), [], 2);    % [T S A L]
    % Get the antenna vector and its length.
    ant_diff = antenna_loc(:, 2:end) - antenna_loc(:, 1); % [3 A-1]
    ant_diff_length = vecnorm(ant_diff); % [1 A-1]
    ant_diff_normalize = ant_diff ./ ant_diff_length; % [3 A-1]
    % Calculate the phase difference.
    phase_diff = csi_phase(:, :, 2:end, :) - csi_phase(:, :, 1, :) - permute(est_rco(2:end, :), [4 3 1 2]); % [T S A-1 L]
    phase_diff = unwrap(phase_diff, [], 2);
    phase_diff = mod(phase_diff + pi, 2 * pi) - pi;
    % Broadcasting is performed, get the value of cos(theta) for each packet and each antenna pair.
    cos_mat = subcarrier_lambda .* phase_diff ./ (2 .* pi .* permute(ant_diff_length, [3 1 2])); % [T S A-1 L]
    cos_mat_mean = squeeze(mean(cos_mat, [2 4])); % [T A-1]
    % Solve the linear equations: ant_diff_normalize' * aoa_sol = cos_mat_mean(p, :)'.
    aoa_mat_sol = ant_diff_normalize' \ cos_mat_mean'; % [3 T]
    % Normalize the result and resolve the singularity.
    % Find the invalid dimensions, where the ant_diff_normalize equals to 0.
    invalid_dim = find(sum(ant_diff_normalize, 2) == 0);
    valid_dim = setdiff([1 2 3], invalid_dim);
    % The value of aoa_mat_sol on the invalid dimension is estimated based on the value on the valid dimention.
    aoa_mat_sol(invalid_dim, :) = repmat(sqrt((1 - sum(aoa_mat_sol(valid_dim, :) .^ 2, 1)) / length(invalid_dim)), 1, length(invalid_dim));
    aoa_mat = aoa_mat_sol;
end
