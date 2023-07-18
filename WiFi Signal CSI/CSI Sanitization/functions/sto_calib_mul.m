function [csi_remove_sto] = sto_calib_mul(csi_src)
    % sto_calib_mul
    % Input:
    %   - csi_src is the csi data with sto; [T S A L]
    % Output:
    %   - csi_remove_sto is the csi data without sto; [T S A L]

    antenna_num = size(csi_src, 3);
    csi_remove_sto = zeros(size(csi_src));
    for a = 1:antenna_num
        a_nxt = mod(a, antenna_num) + 1;
        csi_remove_sto(:, :, a, :) = csi_src(:, :, a, :) .* conj(csi_src(:, :, a_nxt, :));
    end
end