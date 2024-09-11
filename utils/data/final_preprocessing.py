import numpy as np
import scipy.signal
from matplotlib import pyplot as plt

# testing 


def preprocess_EEG(signal,
                   fs=128,
                   stft_size=256,
                   stft_stride=16,
                   lowcut=0.5,
                   highcut=24,
                   visualize=False,
                   labels=None,
                   plot_artifacts=False):
    if visualize:
        # Select random epoch
        rdm_epoch_idx = np.random.randint(2, len(signal) / 4 / fs - 2)
        rdm_epoch_labels = labels.to_numpy()[rdm_epoch_idx - 2:rdm_epoch_idx + 3, :]

        if plot_artifacts:
            labels_dict = {0: 'W_art', 1: 'N_art', 2: 'R_art', 3: 'NREM', 4: 'REM', 5: 'WAKE'}
        else:
            labels_dict = {0: 'NREM', 1: 'REM', 2: 'WAKE'}
        rdm_epoch_labels = np.where(rdm_epoch_labels == 1)[1]
        rdm_epoch_labels = [labels_dict[i] for i in rdm_epoch_labels]

        rdm_epoch_signal = signal[(rdm_epoch_idx - 2) * fs * 4: (rdm_epoch_idx + 3) * fs * 4]
        time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, fs * 4 * 5) / fs

        fig, ax = plt.subplots(6, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [60, 1]})
        fig.subplots_adjust(hspace=0.8)
        cax = ax[0, 0]
        cax.plot(time_axis, rdm_epoch_signal)
        cax.vlines(x=np.linspace(time_axis[0] + 4, time_axis[-1] - 4, 4),
                   ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
        cax.set_title('Raw 5 epochs window')
        # cax.set_xlabel('Time (s)')
        cax.set_xticks(np.linspace(time_axis[0], time_axis[-1], 6))
        cax.set_xlim((time_axis[0], time_axis[-1]))
        epoch_labels_ax = cax.twiny()
        epoch_labels_ax.set_xlim(cax.get_xlim())
        epoch_labels_ax.set_xticks(np.linspace(time_axis[0]+ 2, time_axis[-1]-2, 5))
        epoch_labels_ax.set_xticklabels(rdm_epoch_labels)
        epoch_labels_ax.tick_params(length=0)
        ax[0, 1].axis('off')

    # STFT
    f, t, Z = scipy.signal.stft(signal,
                                fs=128,
                                window='hamming',
                                nperseg=stft_size,
                                noverlap=stft_size - stft_stride
                                )

    if visualize:
        cax = ax[1, 0]

        rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
        # time_axis = np.linspace((rdm_epoch_idx-2)*32, (rdm_epoch_idx+3)*32, 32*5)
        time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, 18) / fs

        img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
        cax.set_title('Spectrogram')
        cax.invert_yaxis()
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel('Frequency (Hz.)')
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels([str(f[-1]), str(f[-1] / 2), str(f[0])])
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
        fig.colorbar(img, cax=ax[1, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

    # Bandpass (crop)
    Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]

    if visualize:
        cax = ax[2, 0]

        rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

        img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
        cax.set_title('Bandpass')
        cax.invert_yaxis()
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel('Frequency (Hz.)')
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['24', '12', '0'])
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
        fig.colorbar(img, cax=ax[2, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

    # PSD
    y = np.abs(Z) ** 2

    if visualize:
        cax = ax[3, 0]

        rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

        img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
        cax.set_title('PSD')
        cax.invert_yaxis()
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel('Frequency (Hz.)')
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['24', '12', '0'])
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
        fig.colorbar(img, cax=ax[3, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

    # Log-scale
    y = 10 * np.log10(y)

    if visualize:
        cax = ax[4, 0]

        rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

        img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
        cax.set_title('Log transformation')
        cax.invert_yaxis()
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel('Frequency (Hz.)')
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['24', '12', '0'])
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
        fig.colorbar(img, cax=ax[4, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])

    # Standardize
    y_mean = np.mean(y, axis=1, keepdims=True)
    y_std = np.std(y, axis=1, keepdims=True)

    y = (y - y_mean) / y_std

    if visualize:
        cax = ax[5, 0]

        rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

        img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
        cax.set_title('Standardization')
        cax.invert_yaxis()
        cax.set_xlabel('Time (s)')
        cax.set_ylabel('Frequency (Hz.)')
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['24', '12', '0'])
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
        fig.colorbar(img, cax=ax[5, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])
        plt.show()

    return y


def preprocess_EMG(signal,
                   fs=128,
                   stft_size=256,
                   stft_stride=16,
                   lowcut=0.5,
                   highcut=30,
                   visualize=False,
                   labels=None,
                   plot_artifacts=False):

    if visualize:
        # Select random epoch
        rdm_epoch_idx = np.random.randint(2, len(signal) / 4 / fs - 2)
        rdm_epoch_labels = labels.to_numpy()[rdm_epoch_idx - 2:rdm_epoch_idx + 3, :]

        if plot_artifacts:
            labels_dict = {0: 'W_art', 1: 'N_art', 2: 'R_art', 3: 'NREM', 4: 'REM', 5: 'WAKE'}
        else:
            labels_dict = {0: 'NREM', 1: 'REM', 2: 'WAKE'}
        rdm_epoch_labels = np.where(rdm_epoch_labels == 1)[1]
        rdm_epoch_labels = [labels_dict[i] for i in rdm_epoch_labels]

        rdm_epoch_signal = signal[(rdm_epoch_idx - 2) * fs * 4: (rdm_epoch_idx + 3) * fs * 4]
        time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, fs * 4 * 5) / fs

        fig, ax = plt.subplots(7, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [60, 1]})
        fig.subplots_adjust(hspace=0.8)
        cax = ax[0, 0]
        cax.plot(time_axis, rdm_epoch_signal)
        cax.vlines(x=np.linspace(time_axis[0] + 4, time_axis[-1] - 4, 4),
                   ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
        cax.set_title('Raw 5 epochs window')
        # cax.set_xlabel('Time (s)')
        cax.set_xticks(np.linspace(time_axis[0], time_axis[-1], 6))
        cax.set_xlim((time_axis[0], time_axis[-1]))
        epoch_labels_ax = cax.twiny()
        epoch_labels_ax.set_xlim(cax.get_xlim())
        epoch_labels_ax.set_xticks(np.linspace(time_axis[0]+ 2, time_axis[-1]-2, 5))
        epoch_labels_ax.set_xticklabels(rdm_epoch_labels)
        epoch_labels_ax.tick_params(length=0)
        ax[0, 1].axis('off')

    # STFT
    f, t, Z = scipy.signal.stft(signal,
                                fs=128,
                                window='hamming',
                                nperseg=stft_size,
                                noverlap=stft_size - stft_stride
                                )

    if visualize:
        cax = ax[1, 0]

        rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
        # time_axis = np.linspace((rdm_epoch_idx-2)*32, (rdm_epoch_idx+3)*32, 32*5)
        time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, 18) / fs

        img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
        cax.set_title('Spectrogram')
        cax.invert_yaxis()
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel('Frequency (Hz.)')
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels([str(f[-1]), str(f[-1] / 2), str(f[0])])
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
        fig.colorbar(img, cax=ax[1, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

    # Bandpass (crop)
    Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]

    if visualize:
        cax = ax[2, 0]

        rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

        img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
        cax.set_title('Bandpass')
        cax.invert_yaxis()
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel('Frequency (Hz.)')
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['24', '12', '0'])
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
        fig.colorbar(img, cax=ax[2, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

    # PSD
    y = np.abs(Z) ** 2

    if visualize:
        cax = ax[3, 0]

        rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

        img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
        cax.set_title('PSD')
        cax.invert_yaxis()
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel('Frequency (Hz.)')
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['24', '12', '0'])
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
        fig.colorbar(img, cax=ax[3, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

    # Integration
    y = np.sum(y, axis=0)

    # Stack rows to have 2 dimensions
    y = np.expand_dims(y, axis=0)
    # y = np.repeat(y, eeg_dimensions[0], axis=0)
    y = np.repeat(y, 48, axis=0)

    if visualize:
        cax = ax[4, 0]

        rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

        img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
        cax.set_title('Integration')
        cax.invert_yaxis()
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel('Frequency (Hz.)')
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['24', '12', '0'])
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
        fig.colorbar(img, cax=ax[4, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])

    # Log-scale
    y = 10*np.log10(y)

    if visualize:
        cax = ax[5, 0]

        rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

        img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
        cax.set_title('Log transformation')
        cax.invert_yaxis()
        # cax.set_xlabel('Time (s)')
        cax.set_ylabel('Frequency (Hz.)')
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['24', '12', '0'])
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
        fig.colorbar(img, cax=ax[5, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])

    # Standardize
    y_mean = np.mean(y, axis=1, keepdims=True)
    y_std = np.std(y, axis=1, keepdims=True)

    y = (y - y_mean) / y_std

    if visualize:
        cax = ax[6, 0]

        rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]

        img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
        cax.set_title('Standardization')
        cax.invert_yaxis()
        cax.set_xlabel('Time (s)')
        cax.set_ylabel('Frequency (Hz.)')
        cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
        cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
        cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
        cax.set_yticklabels(['24', '12', '0'])
        cax.vlines(x=cax.get_xticks()[1:-1],
                   ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
        fig.colorbar(img, cax=ax[6, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])
        plt.show()

    return y


