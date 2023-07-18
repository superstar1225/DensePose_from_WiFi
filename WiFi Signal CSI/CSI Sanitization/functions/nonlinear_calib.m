function [csi_remove_nonlinear] = nonlinear_calib(csi_calib, csi_calib_template)
    % nonlinear_calib
    % Input:
    %   - csi_src is the raw csi which needs to be calibrated; [T S A L]
    %   - csi_calib_template is the reference csi for calibration; [1 S A L]
    % Output:
    %   - csi_remove_nonlinear is the csi data in which the nonlinear error has been eliminated; [T S A L]

    csi_amp = abs(csi_calib);                       % [T S A L]
    csi_phase = unwrap(angle(csi_calib), [], 2);    % [T S A L]
    csi_unwrap = csi_amp .* exp(1i * csi_phase);     % [T S A L]
    % Broadcasting is performed.
    csi_remove_nonlinear = csi_unwrap ./ csi_calib_template;
end