from wav_data_utils import filter_noise, get_local_frequency
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import numpy as np
import pandas as pd
import soundfile as sf

def fill_unexpected_segmentation_with_null(row):
    """
        The segmentation file is not always correct or at the very least has unexepected values
    """

    col_names = {col: i for i, col in enumerate(row.index)}
    # row = pd.Series(row.values, index=range(row.shape[0]))

    diffs = row.diff()
    weird = diffs<0
    idxs = row[weird].index

    for idx in list(idxs):
        pos = col_names[idx]
        row.iloc[pos-1:pos+1] = np.nan
    return row.copy()

def scale_data(row, method='minmax'):
    row_size = row.shape[0]
    no_na = row.dropna()
    row_size_no_na = no_na.shape[0]
    if method == 'minmax':
        no_na = MinMaxScaler((-1 ,1)).fit_transform(no_na.values.reshape(-1, 1)).squeeze()
    elif method == 'standard':
        no_na = StandardScaler().fit_transform(no_na.values.reshape(-1, 1)).squeeze()
    elif method == 'minmaxstandard':
        no_na = MinMaxScaler((-1 ,1)).fit_transform(no_na.values.reshape(-1, 1))
        no_na = StandardScaler().fit_transform(no_na).squeeze()
    else:
        raise 'Invalid method'
    row.iloc[:row_size_no_na] = no_na
    return row

# lets create some noise events 'between' S1/S2
def get_some_noise_data(segmentation_file_df):
    S_1_2_groups = segmentation_file_df.groupby(by=lambda x: x[:2], axis=1).groups
    noise_pos = segmentation_file_df[S_1_2_groups['S1']].values + \
        (segmentation_file_df[S_1_2_groups['S2']].values - segmentation_file_df[S_1_2_groups['S1']].values)/2
    noise_events = pd.DataFrame(noise_pos, index=segmentation_file_df.index,
            columns=['NOISE'] * noise_pos.shape[1])
    noise_events = noise_events.applymap(lambda x: x if np.isnan(x) else int(x))
    return noise_events

class heart_signal(object):
    """
    Create a class to store groups of .wav files
    We can convert data into a dataframe and perform lowpass filtering, perform smoothing, mapping S1/S2 locations
    """

    def __init__(
        self, folder_path=None, files=None, autoread_files=False,
        training_segmentation_file=None, scale_data=None
    ):
        """
        when initialized just store the location of all the data files containing .wav files to read in
        """
        self.provided_files = []
        self.autoread_files = autoread_files
        if not(folder_path is None):
            for r, dlist, flist in os.walk(folder_path):
                """
                read in all files provided by folder
                """
                for f in flist:
                    if not(f.endswith('.wav')):
                        continue
                    name = os.path.join(r, f)
                    self.provided_files.append(name)
        if not (files is None):
            for name in files:
                if os.path.isfile(name) and name.endswith('.wav'):
                    self.provided_files.append(name)
                else:
                    print('Invalid file {0}'.format(name))

        if len(self.provided_files) == 0:
            raise Exception

        if self.autoread_files:
            self.create_df()
        else:
            self.signal_df = None

        if training_segmentation_file:
            self.segmentation_file = pd.read_csv(
               training_segmentation_file,
                index_col=0
            )

            self.segmentation_file.set_index(
               self.segmentation_file.index.map(lambda x: '.'.join(x.split('.')[:-1])),
                inplace=True
            )

        else:
            self.segmentation_file = None

        self.nan_free = None
        self.s1 = None
        self.s2 = None
        self.noise_df = None
        self.scale_data = scale_data


    def create_df(self):
        """
            Read the .wav data into a dataframe where we collect the signal/sound data at the provided sampling rate
        """
        _, self.samplerate = sf.read(self.provided_files[0])

        signal_array = []
        data = []
        for name in self.provided_files:

            signal, samplerate = sf.read(name)
            signal_array.append(list(signal))
            data.append({
                'name': name,
                'index': '__'.join('.'.join(os.path.basename(name).split('.')[:-1]).split('__')[1:]),
                'basename': os.path.basename(name),
                'samplerate': samplerate
            })

        self.data_df = pd.DataFrame(data)
        self.signal_df = pd.DataFrame(signal_array, index=self.data_df['index'])

        if len(self.data_df['samplerate'].unique()) != 1:
            print('The sample rate seems to be different for the provided files!')
        else:
            # we can actually assign "time data" to the columns of the dataframe
            # the "time" will be given in intervals defined by the sampling rate
            time_values = np.arange(0, (self.signal_df.shape[1]*1.0)/self.samplerate, self.samplerate)
            # self.signal_df.rename(columns=lambda x: (1.0 * x)/self.samplerate, inplace=True)

        if not(self.segmentation_file is None):
            # check for unexpected values
            too_large = (self.segmentation_file>self.signal_df.columns[-1]).sum().sum()
            # change any values greater than the number of frames (columns) to null
            self.segmentation_file[self.segmentation_file>self.signal_df.columns[-1]] = np.nan
            print('converted {0} values with unexpected large values to null'.format(too_large))
            not_s1_s2_pattern = (self.segmentation_file.diff(axis=1)<=0).sum().sum()
            if not_s1_s2_pattern > 0:
                self.segmentation_file = self.segmentation_file.apply(
                    lambda x: fill_unexpected_segmentation_with_null(x),
                    axis=1
                )
                print('converted {0} values with unexpected TIME POINTS (i.e. S2 < s1 to null)'.format(not_s1_s2_pattern*3))

        self.filtered_signal_df = self.signal_df
        if self.scale_data:
            self.filtered_signal_df = self.filtered_signal_df.apply(
                lambda x: scale_data(x, self.scale_data),
                axis=1
            )
            # print(self.filtered_signal_df.max(axis=1))
            # print(self.filtered_signal_df.min(axis=1))
        self.data_df.set_index('index', inplace=True)

    def get_signal_length_distribution(self, nbins=None):
        if self.signal_df is None:
            print('You must create_df() first!')
            return
        distanc
        if nbins:
            return np.histogram(
                self.signal_df.apply(lambda x: len(x.dropna())/self.samplerate, axis=1)
            )
        else:
            return np.histogram(self.signal_df.apply(lambda x: len(x.dropna()), axis=1))

    def guess_noise_data(self):
        self.noise_df = get_some_noise_data(self.segmentation_file)

    def get_spectrogram_for_defined_responses(
        self, window_sizes, responses=['S1', 'S2', 'NOISE'], freq_res=1, max_freq=2000
    ):
        """
        Given the signal datframe (each wav file is a row) and the dataframe summarizing where each S1/S2 event
        Calculate a local frequency of amplitudes within a defined window size
        """
        # group column names based on 'signal type'=
        tmp_responses = self.segmentation_file

        S1_S2_col_names = tmp_responses.groupby(by=lambda x: x.split('.')[0], axis=1).groups
        freq_arr_points = np.arange(0, max_freq, freq_res)

        freq_data = []
        spec_data = {}

        for r in responses:
            if r == 'NOISE':
                subset = self.noise_df.stack()
            else:
                subset = tmp_responses[S1_S2_col_names[r]].stack()
            # Get spectrogram for a resonse (S1, S2, NOISE)
            for (file_name, col), pos in subset.iteritems():
                if file_name not in spec_data:
                    spec_data[file_name] = {}
                window = window_sizes.loc[file_name]
                data_point = int(pos)
                samplerate = self.data_df.loc[file_name].samplerate
                signal = self.filtered_signal_df.loc[file_name].values
                tmparr = get_local_frequency(
                    signal, data_point, window, samplerate=samplerate, normalize=False
                )

               # tmparr = tmparr/tmparr[~np.isnan(tmparr)].sum() #np.max(tmparr[~np.isnan(tmparr)])

                tmparr = tmparr.loc[tmparr.index.intersection(freq_arr_points)]


                spec_data[file_name][data_point] = tmparr

                freq_data.append(
                    {
                        'basename': file_name,
                        'response': r,
                        'col_name': col,
                        'pos': pos,
                        'window': window
                    }
                )

        tmp = pd.Panel(spec_data).to_frame(filter_observations=False).T.stack()
        # gr = tmp.groupby(lambda x: np.ceil(x/1000), axis=1).apply(lambda x: x.sum(axis=1))

        tmpinfo = pd.DataFrame(freq_data)
        tmpinfo2 = tmpinfo.groupby(by=['basename', 'pos']).filter(lambda x: len(x) == 1)
        if tmpinfo2.shape[0] < tmpinfo.shape[0]:
            print('WARNING DUPLICATE ROWS WERE DELETED, UNEXPECTED!!!')
            print(tmpinfo.groupby(by=['basename', 'pos']).filter(lambda x: len(x)>1))
        return pd.concat(
            [
                tmpinfo2.set_index(['basename', 'pos']),
                tmp
            ],
            keys=['info', 'spec'],
            axis=1
        )

    def get_spectrogram_at_all_points(
        self, window_sizes, freq_res=1, max_freq=2000
    ):
        """
        Given the signal datframe (each wav file is a row),
        calculate a local frequency of amplitudes at all points in the wav files
        """
        freq_arr_points = np.arange(0, max_freq, freq_res)

        freq_data = []
        spec_data = {}


        for file_name, signal in self.filtered_signal_df.iterrows():
            print(file_name)
            if file_name not in spec_data:
                spec_data[file_name] = {}
            window = window_sizes.loc[file_name]
            samplerate = self.data_df.loc[file_name].samplerate

            signal = signal.dropna().values
            for data_point in range(len(signal)):
                tmparr = get_local_frequency(
                    signal, data_point, window, samplerate=samplerate, normalize=False
                )
                # tmparr = tmparr/tmparr[~np.isnan(tmparr)].sum() #np.max(tmparr[~np.isnan(tmparr)])
                spec_data[file_name][data_point] = tmparr

                freq_data.append(
                    {
                        'basename': file_name,
                        'pos': data_point,
                        'window': window
                    }
                )



        tmp = pd.Panel(spec_data).to_frame(filter_observations=False).T.stack()
        tmpinfo = pd.DataFrame(freq_data)

        return pd.concat(
            [
                tmpinfo.set_index(['basename', 'pos']),
                tmp
            ],
            keys=['info', 'spec'],
            axis=1
        )

    def identify_null_values(self):
        """
        It makes sense for the data to have NaN at the end of the signal. This is because each .wav file is a different
        length. But not sure if there is any NaN data in the middle of the file. So this makes sure there is only
        one spoint in the data that has Nan values (i.e. doesnt do 0 NaN 1 Nan 2 Nan Nan; but it can do 0 1 2 NaN NaN NaN)
        """
        check_for_nan = (self.signal_df.isnull().diff(axis=1).iloc[:, 1:] == True).sum(axis=1)
        has_unusual_data = check_for_nan[check_for_nan>1]
        if has_unusual_data.shape[0] > 0:
            self.nan_free = False
            print('WE found files with NaN in them')
        else:
            self.nan_free = True
            print('Files look good (no NaN in the middle)')
        return has_unusual_data.index

    def get_some_data(self, index, use_iloc=False):
        """
        return a copy of the signals
        """
        if use_iloc:
            return self.signal_df.iloc[index].copy()
        else:
            return self.signal_df.loc[index].copy()


    def pad_zero(self, check_nan_free = True):
        if check_nan_free and (self.nan_free is None or check_nan_free is False):
            print('Sorry I cannot do this function until I know all nan values have been handled correctly!')
            raise Exception
            return

        if self.nan_free is None:
            print("Warning: Im replacing all NaN points in the table with zeros but I havent checked if there are any Nan values in the middle of the dataset...this could screw up the results")
        elif self.nan_free is False:
            print('Warning: Im replacing all Nan points in the table with zeros but some datapoints with Nan are not at the end of the signal and might yield incorrect results')

        self.signal_df.fillna(0, inplace=True)

    def low_pass_filter(self, freq, limit=None, inplace=False):
        """
        Applies a low pass filter to the signal dataframe

        freq: low-pass filter cutoff

        returns:
            df (signal df)
        """
        if limit:
            subsample = self.signal_df.iloc[:limit]
        else:
            subsample = self.signal_df

        new_data = [filter_noise(d, freq, self.samplerate) for d in subsample.values]

        new_data_df = pd.DataFrame(new_data, index=subsample.index)
        time_values = np.arange(0, (new_data_df.shape[1]*1.0)/self.samplerate, self.samplerate)
        # new_data_df.rename(columns=lambda x: (1.0 * x)/self.samplerate, inplace=True)

        if inplace:
            self.filtered_signal_df = new_data_df
            # print('OK!')
            print(self.scale_data)
            if self.scale_data:
                self.filtered_signal_df = self.filtered_signal_df.apply(
                    lambda x: scale_data(x, self.scale_data),
                    axis=1
                )
                # print(self.filtered_signal_df.max(axis=1))
                # print(self.filtered_signal_df.min(axis=1))
            # print('YAD!')
        else:
            return new_data_df

    def heart_beat_rate_to_hz(self, rate_per_min=60):
        # rate_per_min = total (lub/dub) beats per minute
        beats_per_sec = (rate_per_min*1.0)/60
        return beats_per_sec

    def heart_beat_rate_to_sampling_rate(self, rate_per_min=60):
        """
        Given a heart beat rate, return the number of "frames" in
        the wav file that captures a single beat within a window size
        """
        # rate_per_min = total (lub/dub) beats per minute
        beat_hz = self.heart_beat_rate_to_hz(rate_per_min)
        # samplerate => frames/second
        # beat_hz => beats/secon, inverse = seconds/beats
        # frames_per_heartbeat = (frames/second) * (second/beats) = frames/second
        frames_per_heartbeat = self.samplerate*(1/beat_hz)
        return frames_per_heartbeat

    def spectrogram(self, window_size, noverlap=None, additional_kwargs={},
        limit=None, randomly_select=None, keep_time_unit=False,
        use_filtered_data = True):

        random_signals = self.filtered_signal_df if use_filtered_data else self.signal_df

        if randomly_select:
            random_signals.subsample(randomly_select)

        if limit:
            limit = min(limit, random_signals.shape[0])
            random_signals = random_signals.iloc[:limit]

        freqs_sampled, average_time_window, Sxx = signal.spectrogram(
            random_signals.values, self.samplerate,
            nperseg=int(window_size),
            noverlap=int(noverlap),
            **additional_kwargs
        )

        if keep_time_unit is False:
            average_time_window *= np.round(self.samplerate)

        return freqs_sampled, average_time_window, \
                pd.Panel(Sxx, items=random_signals.index, major_axis=freqs_sampled, minor_axis=average_time_window)

    def return_S1_S2_noise_position(self, inplace=True):
        """
            Given the dataframe where the S1/S2 or lub/dub demarcations have been called out already,
            return the original data where 'S1' is filled in the proper index and 'S2' is filled in the proper indexes
        """

        # isolate only the rows listed in heart_calls_df
        heart_calls_df = self.segmentation_file
        if not(self.noise_df is None):
            heart_calls_df = pd.concat([heart_calls_df, self.noise_df], axis=1)
        subset = self.signal_df.loc[self.signal_df.index.intersection(heart_calls_df.index)]
        # make a dummy dataframe to store results
        result_calls = pd.DataFrame(np.nan, index=subset.index, columns=subset.columns)
        df = heart_calls_df.loc[self.signal_df.index.intersection(heart_calls_df.index)]

        # group column names based on 'signal type'
        S1_S2_col_names = heart_calls_df.groupby(by=lambda x: x.split('.')[0], axis=1).groups

        # isolate the 'S1' columns from the test set (lub)
        s1_columns = S1_S2_col_names['S1']

        # isolate the 'S2' columns from the test set (dub)
        s2_columns = S1_S2_col_names['S2']

        noise_columns = S1_S2_col_names['NOISE']

        # really slow and hacky code, but go through each row and find the frames that refer to S1 or S2
        # LABEL THAT ROW/COLUMN COMBINATION in the dummy dataframe
        for i, (r, v) in enumerate(df.iterrows()):
            s1_col = v.loc[s1_columns].dropna().values
            result_calls.ix[i, s1_col] = 'S1'
            s2_col = v.loc[s2_columns].dropna().values
            result_calls.ix[i, s2_col] = 'S2'
            noise_col = v.loc[noise_columns].dropna().values
            result_calls.ix[i, noise_col] = 'N'

        if inplace:
            self.S1_S2_demarcation = result_calls
        else:
            return result_calls

    def return_event_position(self, heart_calls_df, lambda_fxn=lambda x: True, label='label'):
        """
            Given the dataframe where the S1/S2 or lub/dub demarcations have been called out already,
            return the original data where 'S1' is filled in the proper index and 'S2' is filled in the proper indexes
        """

        # isolate only the rows listed in heart_calls_df
        subset = self.signal_df.loc[self.signal_df.index.intersection(heart_calls_df.index)]
        # make a dummy dataframe to store results
        result_calls = pd.DataFrame(np.nan, index=subset.index, columns=subset.columns)
        df = heart_calls_df.loc[self.signal_df.index.intersection(heart_calls_df.index)]

        # isolate the columns which return true for the lambda_fxn
        return_columns = [s for s in df.columns if lambda_fxn(s)]

        # really slow and hacky code, but go through each row and find the frames that refer to the lambda_fxn
        # LABEL THAT ROW/COLUMN COMBINATION in the dummy dataframe
        for i, (r, v) in enumerate(df.iterrows()):
            s1_col = v[return_columns].dropna().values
            result_calls.ix[i, s1_col] = label

        return result_calls

    def bin_signal(self, nperseg, noverlap, fs=1.0):
        """
            given a window, return the indicies of the dataframe where data should be binned into it
        """
        col_as_array = self.signal_df.columns
        skip = int(nperseg)
        overlap=int(noverlap)
        bins = OrderedDict()
        for c in np.arange(
            0,
            col_as_array.max(),
            skip - overlap
        ):
            p1 = c
            p2 = (c + skip)
            bins[(c + (skip)/2.0)/fs] = list(col_as_array[p1:p2])
        return bins

    def get_average_signal_in_bin(self, nperseg, noverlap, fs=1.0, method='mean'):
        binned_data = self.bin_signal(nperseg, noverlap, fs)
        x_vals = np.array(binned_data.keys())
        new_df = {}
        for k, val in binned_data.iteritems():
            if method == 'mean':
                new_df[k] = self.filtered_signal_df[val].mean(axis=1)
            elif method == 'max':
                # returns either the maximum or minimum value in a group
                new_df[k] = self.filtered_signal_df[val].apply(
                    lambda x: x[np.abs(x.dropna()).idxmax()] if len(x.dropna())>0 else np.nan,
                    axis=1,
                )

            else:
                print('performing median')
                new_df[k] = self.filtered_signal_df[val].median(axis=1)
        return pd.DataFrame(new_df)

    def get_window_signal(self, windows, method='sum', absV=True):
        # signal_df = self.filtered_signal_df.apply(
        #    lambda x: StandardScaler().fit_transform(x.fillna(0).values.reshape(-1, 1)).squeeze(),
        #    axis=1
        #).abs()
        # print(windows.mean())

        if absV:
            signal_df = self.filtered_signal_df.apply(lambda x: np.abs(x), axis=1)
        else:
            signal_df = self.filtered_signal_df.copy()


        merged_data = pd.concat(
            [
                signal_df,
                windows
            ], axis=1, keys=['signal', 'window']
        )

        if method == 'sum':
            return merged_data.apply(
                lambda x: x['signal'].rolling(int(x['window']), center=True).apply(lambda x: x.sum()),
                axis=1
            )
        elif method == 'mean':
            return merged_data.apply(
                lambda x: x['signal'].rolling(int(x['window']), center=True).apply(lambda x: x.mean()),
                axis=1
            )
        elif method == 'max':
            return merged_data.apply(
                lambda x: x['signal'].rolling(int(x['window']), center=True).apply(lambda x: x.max()),
                axis=1
            )
        elif method == 'min':
            return merged_data.apply(
                lambda x: x['signal'].rolling(int(x['window']), center=True).apply(lambda x: x.min()),
                axis=1
            )
        elif method == 'median':
            return merged_data.apply(
                lambda x: x['signal'].rolling(int(x['window']), center=True).apply(lambda x: np.median(x)),
                axis=1
            )


    def distance_from(self, windows, method='max'):
        signal_df = self.filtered_signal_df.copy()
        # print(windows.mean())
        merged_data = pd.concat(
            [
                signal_df,
                windows
            ], axis=1, keys=['signal', 'window']
        )

        if method == 'max':
            peak_pos = merged_data.apply(
                lambda x: x['signal'].rolling(int(x['window']), center=True).apply(lambda x: np.argmax(x)),
                axis=1
            )
        elif method == 'min':
            peak_pos = merged_data.apply(
                lambda x: x['signal'].rolling(int(x['window']), center=True).apply(lambda x: np.argmin(x)),
                axis=1
            )
        else:
            raise 'InvalidMethod'

        return merged_data['signal'] - peak_pos


    def return_s1_s2_binned_data(self, s1_s2_df, nperseg, noverlap, fs=1.0):
        binned_data = self.bin_signal(nperseg, noverlap, fs)
        s1_s2_bins = self.return_S1_S2_position(s1_s2_df, inplace=False)

        new_df = {}
        for k, val in binned_data.iteritems():
            new_df[k] = s1_s2_bins[val].apply(
                lambda x: np.array(','.join(sorted(list(x.dropna().unique())))) if len(x.dropna()) > 0 else '',
                axis=1
            )
        return pd.DataFrame(new_df)

    def get_average_freq_contribution(self, frame_cutoff=1000, window=100):
        """
            1. Convert signal to frequency contributes
            2. Only consider frequencies within the first [:frame_cutoff] indexes
            3. Smooth out the signal/get an average coefficient of frequencies defined by the window
            4. only return coefficients in multiples of the window
        """

        def smoothed_coefficients(signal):
            fft_signal = fftpack.fft(signal.dropna())
            fft_freqs = fftpack.fftfreq(fft_signal.shape[0], 1.0/self.samplerate)
            lim = range(frame_cutoff)
            abs_sign = pd.Series(np.abs(fft_signal[lim]), index=fft_freqs[lim])
            abs_sign =  abs_sign.rolling(window=window).mean().iloc[range(0, abs_sign.shape[0], window)].dropna()
            return abs_sign/abs_sign.max()

        return self.filtered_signal_df.apply(smoothed_coefficients, axis=1).fillna(0)


def trim_dataframe_by_time(df, samplerates, total_time, l_time_trim, r_time_trim):
    """
    Remove the start and end of a wav file (usually is very noisy because of activity)
    """
    a = []
    for i, vals in df.iterrows():
        rate = samplerates.loc[i]
        cutoff = int(rate*1.0*l_time_trim)
        last_time_point = total_time.loc[i].dropna().shape[0]
        r_cutoff = int(rate*1.0*r_time_trim)
        vals.loc[:cutoff] = np.nan
        vals.loc[last_time_point-r_cutoff:] = np.nan
        a.append(vals)
    return pd.DataFrame(a)


def load_wav_data_into_features(
    wave_folder_path, training_segmentation_file, noise_filter=50, freq_amplitude_cutoffs=0.05, time_point_avg_window=0.1,
    time_point_spec=0.1, l_time_trim = 1, r_time_trim = 1, normalize_raw_signal=None
):
    """
        Automate steps for extracting features from a set of WAV files
    """
    print('settings')
    print('noise filter', noise_filter)
    print('window_amplitudes', time_point_avg_window)
    print('window_spec', time_point_spec)
    print('trims:', l_time_trim, r_time_trim)
    print('normalization', normalize_raw_signal)

    # 1: Instantiate the class. Point it to where the wav files exist
    print('reading in files')
    heart_group = heart_signal(
        folder_path=wave_folder_path, training_segmentation_file=training_segmentation_file,
        scale_data=normalize_raw_signal
    )

    # 2. Load the wav files into a dataframe
    heart_group.create_df()
    print(heart_group.data_df.samplerate.mean())
    # 3. Using the segmentation file, also add in noise data points
    heart_group.guess_noise_data()
    # 4. filter out noisy data
    if noise_filter:
        print('Low pass filter')
        heart_group.low_pass_filter(noise_filter, inplace=True)
    # 5. Analyze the frequency signals from the data . Using the frequencies, create a window size based on amplitudes
    # print('Guessing frequencies')
    # smoothed_freq_df = heart_group.filtered_signal_df.apply(
    #    lambda x: get_smoothed_arr_df(x, heart_group.data_df.loc[x.name].samplerate), axis=1
    #)

    # return heart_group
    # smoothed_freq_df[smoothed_freq_df < freq_amplitude_cutoffs] = 0
    # smoothed_freq_df.fillna(0, inplace=True)
    # this returns possible windows for each wav file using the weighted distrbution of amplitudes at each frequency
    # windows = smoothed_freq_df.apply(lambda x: int(sum(np.array(x.index)*x.values))*2 ,axis=1)
    # windows = pd.Series(500, index=heart_group.data_df.index)
    # print(heart_group.samplerate)
    # print(heart_group.data_df)
    windows = heart_group.data_df.samplerate.apply(lambda x: int(x*1.0*time_point_avg_window))
    # print(windows.mean())
    # 6. using the calculated windows sizes, get the average amplitude in a window
    print('Get average amplitude')
    max_abs = heart_group.get_window_signal(windows, method='max')
    sum_abs = heart_group.get_window_signal(windows, method='sum')
    max_signal = heart_group.get_window_signal(windows, method='max', absV=False)
    min_signal = heart_group.get_window_signal(windows, method='min', absV=False)
    distance_from_max = heart_group.distance_from(windows, method='max')
    distance_from_min = heart_group.distance_from(windows, method='min')

    #for w in [0.001, 0.01, 0.1, 1, 2]:
    #    windows = heart_group.data_df.samplerate.apply(lambda x: int(x*1.0*w))
    #    heart_group.get_window_signal(windows, method='sum').to_csv('window_amplitudes_' + str(w) + '.csv')

    print('individual amplitude')
    amplitudes = heart_group.filtered_signal_df.abs()
    print('individual differences')
    # get "Differneces" in raw amplitdue
    amplitude_difference = heart_group.filtered_signal_df.rolling(window=3, center=True, axis=1).apply(lambda x: (x[-1] - x[0])/3)
    # 7. Also get spectrograms
    print('Get spectrogram')
    windows = heart_group.data_df.samplerate.apply(lambda x: int(x*1.0*time_point_spec))
    spectrogram_features = heart_group.get_spectrogram_for_defined_responses(
        windows, responses=['S1', 'S2', 'NOISE'],
    )
    specs = {}

    # 9. Generate features from data
    print('generate features')
    feature_amplitudes = sum_abs  # average amplitude in a position/window
    feature_max_abs = max_abs
    feature_max_amp = max_signal
    feature_min_amp = min_signal
    feature_ind_amplitudes = amplitudes
    feature_dis_max = distance_from_max
    feature_dis_min = distance_from_min
    feature_amplitude_changes = amplitude_difference  # differnces in average amplitudes in a poisitoin/window


    S1_S2_and_noise_datapoints = heart_group.return_S1_S2_noise_position(inplace=False)

    S1_S2_and_noise_datapoints = trim_dataframe_by_time(
        S1_S2_and_noise_datapoints,
        heart_group.data_df.samplerate,
        heart_group.filtered_signal_df,
        l_time_trim,
        r_time_trim
    )

    feature_amplitudes[S1_S2_and_noise_datapoints.isnull()] = np.nan
    feature_max_abs[S1_S2_and_noise_datapoints.isnull()] = np.nan
    feature_max_amp[S1_S2_and_noise_datapoints.isnull()] = np.nan
    feature_min_amp[S1_S2_and_noise_datapoints.isnull()] = np.nan
    feature_dis_max[S1_S2_and_noise_datapoints.isnull()] = np.nan
    feature_dis_min[S1_S2_and_noise_datapoints.isnull()] = np.nan
    feature_amplitude_changes[S1_S2_and_noise_datapoints.isnull()] = np.nan
    feature_ind_amplitudes[S1_S2_and_noise_datapoints.isnull()] = np.nan

    # spectrogram_features['spec'] => the spectrogram at each S1/S2/NOISE position
    useful_frequencies = spectrogram_features['spec'].loc[:, spectrogram_features['spec'].std(axis=0)>0]
    # spectrogram_features['info'] 'REPORTS WHETHER POSITION IS s1/s2/noise and where that position occurred and window size
    # final features dataframe (y = spectrogram_features['info']['response'])
    features_and_response = pd.concat(
        [
            spectrogram_features['info'],
            useful_frequencies,
            feature_amplitudes.stack(),
            feature_max_abs.stack(),
            feature_max_amp.stack(),
            feature_min_amp.stack(),
            feature_dis_max.stack(),
            feature_dis_min.stack(),
            feature_amplitude_changes.stack(),
            feature_ind_amplitudes.stack(),
            # feature_ind_diff.stack()
        ], axis=1, join='inner', keys=[
            'info', 'spec', 'AverageAmplitude', 'MaxAbs', 'MaxAmp', 'MinAmp', 'MaxDist', 'MinDist', 'AmplitudeDifference', 'IndAmp'
        ]
    )
    return features_and_response, heart_group  # , windows, smoothed_freq_df, feature_amplitudes#, window_amplitudes, specs


