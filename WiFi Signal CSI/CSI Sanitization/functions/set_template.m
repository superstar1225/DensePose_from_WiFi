function [csi_calib_template] = set_template(csi_calib, linear_interval, calib_template_name)
    % set_template
    % Input:
    %   - csi_calib is the reference csi at given distance and angle; [T S A L]
    %   - linear_interval is the linear range of the csi phase, which varies across different types of NICs;
    %   - calib_template_name is the saved path of the generated template;
    % Output:
    %   - csi_calib_template is the generated template for csi calibration; [1 S A L]

    [packet_num, subcarrier_num, antenna_num, extra_num] = size(csi_calib);
    csi_amp = abs(csi_calib);                       % [T S A L]
    csi_phase = unwrap(angle(csi_calib), [], 2);    % [T S A L]
    csi_amp_template = mean(csi_amp ./ mean(csi_amp, 2), 1); % [1 S A L]
    nonlinear_phase_error = zeros(size(csi_calib));          % [T S A L]
    for p = 1:packet_num
        for a = 1:antenna_num
            for e = 1:extra_num
                linear_model = fit(linear_interval, squeeze(csi_phase(p, linear_interval, a, e))', 'poly1');
                nonlinear_phase_error(p, :, a, e) = csi_phase(p, :, a, e) - linear_model(1:subcarrier_num)';
            end
        end
    end
    csi_phase_template = mean(nonlinear_phase_error, 1); % [1 S A L]
    csi_phase_template(1, linear_interval, :, :) = 0;
    csi_calib_template = csi_amp_template .* exp(1i * csi_phase_template); % [1 S A L]
    csi = csi_calib_template;
    save(calib_template_name, 'csi'); % [1 S A L]
end