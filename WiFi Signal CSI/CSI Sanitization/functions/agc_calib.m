function [csi_remove_agc] = agc_calib(csi_src, csi_agc)
    % rco_calib
    % Input:
    %   - csi_src is the raw csi which needs to be calibrated; [T S A L]
    %   - csi_agc is the AGC amplitude reported by the NIC; [T, 1]
    % Output:
    %   - csi_remove_agc is the csi data in which the AGC uncertainty has been eliminated; [T S A L]

    % Broadcasting is performed.
    csi_remove_agc = csi_src ./ csi_agc;
end