function [intrusion_flag] = naive_intrusion(csi_data, threshold)
    % naive_intrusion
    % Input:
    %   - csi_data is the CSI used for intrusion detection; [T S A L]
    %   - threshold determines the sensitivity, and the optimal threshold setting varies with the environment.
    % Output:
    %   - intrusion_flag indicates whether there's intrusion in the indoor environment (True) or not (False);

    [packet_num, subcarrier_num, antenna_num, extra_num] = size(csi_data);
    csi_data_normalize = csi_data / mean(abs(csi_data), 'all');
    std_collection = zeros(subcarrier_num, antenna_num, extra_num);
    parfor s = 1:subcarrier_num
        for a = 1:antenna_num
            for e = 1:extra_num
                cur_fft = abs(fft(csi_data_normalize(:, s, a, e)));
                std_collection(s, a, e) = mean(cur_fft(2:50));
            end
        end
    end
    intrusion_indicator = mean(rmoutliers(std_collection(:)), 'all');
    intrusion_flag = (intrusion_indicator > threshold);
end