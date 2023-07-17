function stft_mat = naive_spectrum(csi_data, sample_rate, visable)
    % naive_spectrum
    % Input:
    %   - csi_data is the CSI used for STFT spectrum generation; [T S A L]
    %   - sample_rate determines the resolution of the time-domain and
    %   frequency-domain;
    % Output:
    %   - stft_mat is the generated STFT spectrum; [sample_rate/2 T]

    % Conjugate multiplication.
    csi_data = mean(csi_data .* conj(csi_data), [2 3 4]);
    % Calculate the STFT and visualization.
    stft_mat = stft(csi_data, sample_rate);
    % Visualization (optional).
    if visable
        stft(csi_data, sample_rate);
    end
end