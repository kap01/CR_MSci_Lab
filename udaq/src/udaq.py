import numpy as np
import sys
import time
from pathlib import Path
import configparser
import csv
from math import floor

from picoscope_6000a import PicoScope6000A

class udaq():

    def __init__(self):

        self._CHANNELS = ['A', 'B', 'C', 'D','E', 'F', 'G', 'H']

        self._num_events = 0
        self._combo_type = 2
        self._coupling = {'A': 'DC', 'B': 'DC', 'C': 'DC', 'D': 'DC','E': 'DC', 'F': 'DC', 'G': 'DC', 'H': 'DC'}
        self._polarity = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1}
        self._is_enabled = {'A': False, 'B': False, 'C': False, 'D': False, 'E': False, 'F': False, 'G': False, 'H': False}
        self._is_trigger_enabled = {'A': False, 'B': False,
                                    'C': False, 'D': False,
                                    'E': False, 'F': False,
                                    'G': False, 'H': False}
        self._trigger_type = {'A': 'LEVEL', 'B': 'LEVEL',
                              'C': 'LEVEL', 'D': 'LEVEL',
                              'E': 'LEVEL', 'F': 'LEVEL',
                              'G': 'LEVEL', 'H': 'LEVEL'}
        self._trigger_direction = {'A': 'RISING', 'B': 'RISING',
                                   'C': 'RISING', 'D': 'RISING',
                                   'E': 'RISING', 'F': 'RISING',
                                   'G': 'RISING', 'H': 'RISING'}
        self._range = {'A': 10, 'B': 10, 'C': 10, 'D': 10,
                       'E': 10, 'F': 10, 'G': 10, 'H': 10}
        self._is_baseline_correction_enabled = {'A': False, 'B': False,
                                                'C': False, 'D': False,
                                                'E': False, 'F': False,
                                                'G': False, 'H': False}
        self._threshold = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0,
                           'E': 0.0, 'F': 0.0, 'G': 0.0, 'H': 0.0}
        self._timing = {'A': 'PEAK', 'B': 'PEAK', 'C': 'PEAK', 'D': 'PEAK',
                        'E': 'PEAK', 'F': 'PEAK', 'G': 'PEAK', 'H': 'PEAK'}

        self._timebase = 0
        self._sample_interval = 0
        self._pre_trigger_window = 0.
        self._post_trigger_window = 0.
        self._pre_samples = 0
        self._post_samples = 0
        self._num_samples = 0
        self._num_captures = 0

        self._t_start_run = 0
        self._elapsed_time = 0
        self._run_time = 0

        self._num_runs = 1
        self._output_path = None
        self._run_number = 0
        self._output_filename = None
        self._output_file = None

        if len(sys.argv) == 2:
            self._config_filename = Path.cwd()/str(sys.argv[1])
        else:
            self._config_filename = Path.cwd()/'udaq.config'

        self._read_configuration()

        self.scope = PicoScope6000A()

    def _read_configuration(self):
        print(f'Configuration file: {self._config_filename}')
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(str(self._config_filename))

        path = config['Run']['Output Path']
        if path == '.':
            self._output_path = Path.cwd()
        else:
            self._output_path = Path.cwd()/path
        self._run_time = config.getfloat('Run', 'Run Time')
        if config.has_option('Run','Number of Runs'):
            self._num_runs = config.getint('Run', 'Number of Runs')
        self._combo_type = config.getint('Run', 'Trigger Combo Setting')
        self._pre_trigger_window = config.getfloat('Sampling',
                                                    'Pre-Trigger Window')
        self._post_trigger_window = config.getfloat('Sampling',
                                                    'Post-Trigger Window')
        self._timebase = config.getint('Sampling', 'Time Base')
        self._num_captures = config.getint('Sampling',
                                           'Captures Per Block')
        
        for ch in self._CHANNELS:
            sec = 'Channel ' + ch
            self._is_enabled[ch] = sec in config.sections()
            if self._is_enabled[ch]:
                self._coupling[ch] = config[sec]['Coupling']
                self._polarity[ch] = np.sign(config.getint(sec, 'Polarity'))
                self._range[ch] = config.getfloat(sec, 'Range')
                self._is_baseline_correction_enabled[ch] \
                    = config[sec].getboolean('Baseline Correction')
                self._timing[ch] = config[sec]['Timing']
                self._is_trigger_enabled[ch] \
                    = config[sec].getboolean('Trigger Enabled')
                self._trigger_type[ch] = config[sec]['Trigger Type']
                self._trigger_direction[ch] \
                    = config[sec]['Trigger Direction']
                self._threshold[ch] = config[sec].getfloat('Threshold')

    def _set_channels(self):
        for ch in self._CHANNELS:
            if self._is_enabled[ch]:
                self.scope.set_channel(ch, self._coupling[ch],
                                       self._range[ch], 0)

    def _set_sampling(self):
        dt = self.scope.get_interval_from_timebase(
            self._timebase)
        self._pre_samples  = floor(self._pre_trigger_window / 1e-9 / dt)
        self._post_samples = floor(self._post_trigger_window / 1e-9 / dt) + 1
        self._num_samples  = self._pre_samples + self._post_samples
        self._sample_interval = dt

    def _set_triggers(self):
        self.scope.set_advanced_triggers(self._is_trigger_enabled,
            self._trigger_type, self._trigger_direction, self._threshold, self._combo_type)

    def _open_output_file(self):
        for x in Path(self._output_path).glob('*.csv'):
            if x.is_file() and x.name[0:3] == 'Run':
                self._run_number = int(x.name[3:7]) + 1

        self._output_filename =  self._output_path / 'Run{0:04d}.csv'\
                                                       .format(self._run_number)

        try:
             self._output_file = open(self._output_filename, 'w',
                                      newline='')
        except IOError:
            print('Error: Unable to open: {}'\
                  .format(self._output_filename))
            return 0

        writer = csv.writer(self._output_file)
        header = []
        for ch in self._CHANNELS:
            if self._is_enabled[ch]:
                header.append('time_'+ch)
                header.append('pulse_height_'+ch)
        writer.writerow(header)

    def _process_data(self, data):
        x = data['x']
        times, baselines, pulseheights = [], [], []
        bl_samples = int(self._pre_samples * .8)
        for ch in self._CHANNELS:
            if self._is_enabled[ch]:
                channel_data = data[ch]
                if self._timing[ch] == 'PEAK':
                    ts = x[np.argmax(channel_data*self._polarity[ch], axis=1)]
                else: # ZERO or LEVEL
                      #     (https://stackoverflow.com/questions/23289976/
                      #       how-to-find-zero-crossings-with-hysteresis)
                    hi = channel_data >= self._threshold[ch]
                    if self._timing[ch] == 'ZERO':
                        lo = channel_data <= -self._threshold[ch]
                    else: # lEVEL
                        lo = channel_data <= self._threshold[ch]
                    if self._trigger_direction[ch] == 'FALLING':
                        y = hi[:, 1:]<hi[:, :-1]
                        z = lo[:, 1:]>lo[:, :-1]
                    else: # RISING
                        y = lo[:, 1:]<lo[:, :-1]
                        z = hi[:, 1:]>hi[:, :-1]
                    # Place a True at the end of the sampling period to handle
                    # failure to find a crossing.
                    fail = np.zeros(np.shape(y), dtype=bool)
                    fail[:,-1] = True
                    into  = np.where(y, y, fail)
                    outof = np.where(z, z, fail)
                    # Index of the *first* True (fail = last).
                    i_in = np.argmax(into, axis=1)
                    i_out = np.argmax(outof, axis=1)
                    ts = x[i_out]
                times.append(ts)
                channel_data *= self._polarity[ch]
                if self._is_baseline_correction_enabled[ch] and bl_samples > 0:
                    bl = channel_data[:, :bl_samples].mean(axis=1)
                else:
                    bl = np.zeros(len(channel_data))
                ph = (channel_data.max(axis=1) - bl)
                pulseheights.append(ph)
        times = np.array(times)
        return np.array(times), np.array(pulseheights)

    def _write_output(self, times, pulseheights):
        writer = csv.writer(self._output_file, quoting=csv.QUOTE_NONE)
        table = []
        for i in range(len(times)):
            if self._is_enabled[self._CHANNELS[i]]:
                table.append(times[i])
                table.append(pulseheights[i])
        for row in zip(*table):
            writer.writerow(row)

    def _close_output_file(self):
        try:
            self._output_file.close()
        except IOError:
            print('Error: Unable to close: {}'\
                  .format(self._output_filename))

    def _write_info_file(self):
        info_filename = self._output_filename.with_suffix('.info')
        try:
            info_file = open(info_filename, 'w', newline='', encoding="utf-8")
        except IOError:
            print(f'Error: Unable to open: {info_filename}\n')
        info_file.write(f'# Configuration File: {str(self._config_filename)}\n')
        info_file.write('\n[Run]\n')
        info_file.write(f'Start Time: {time.ctime(self._t_start_run)}\n')
        info_file.write(f'Run Time (s): {self._elapsed_time:.1f}\n')
        info_file.write(f'Events: {self._num_events}\n')

        info_file.write('\n[Sampling]\n')
        info_file.write('Pre-Trigger Window (s): {0:.2e}\n'\
                        .format(self._pre_trigger_window))
        info_file.write('Post-Trigger Window (s): {0:.2e}\n'\
                        .format(self._post_trigger_window))
        info_file.write(f'Time Base: {self._timebase}\n')
        info_file.write(f'Sample Interval (ns): {self._sample_interval}\n')
        info_file.write(f'Samples Per Capture: {self._num_samples}\n')
        info_file.write(f'Captures Per Block: {self._num_captures}\n')

        for ch in self._CHANNELS:
            if self._is_enabled[ch]:
                info_file.write(f'\n[Channel {ch}]\n')
                info_file.write(f'Coupling: {self._coupling[ch]}\n')
                info_file.write(f'Polarity: {self._polarity[ch]}\n')
                info_file.write(f'Range: {self._range[ch]}\n')
                info_file.write('Baseline Correction: {0}\n'\
                    .format(self._is_baseline_correction_enabled[ch]))
                info_file.write(f'Timing: {self._timing[ch]}\n')
                info_file.write('Trigger Enabled: {0}\n'\
                    .format(self._is_trigger_enabled[ch]))
                info_file.write('Trigger Type: {0}\n'\
                    .format(self._trigger_type[ch]))
                info_file.write('Trigger Direction: {0}\n'\
                    .format(self._trigger_direction[ch]))
                info_file.write('Threshold (V): {0:.4f}\n'\
                    .format(self._threshold[ch]))

        info_file.close()

    def _acquire_run(self):

        self._open_output_file()

        print(f'\nRun{self._run_number:04d}')

        self._t_start_run = time.time()
        self._elapsed_time = 0
        self._num_events = 0
        while time.time() - self._t_start_run < self._run_time:
            self.scope.start_run(self._pre_samples, self._post_samples,
                                 self._timebase, self._num_captures)    
            while(self.scope.wait_for_data() == 0):
                self._elapsed_time = time.time() - self._t_start_run
                if self._elapsed_time >= self._run_time:
                    self._close_output_file()
                    self._write_info_file()
                    return 0
            else:
                self._num_events += self._num_captures
                keys = ['x', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                x, [A, B, C, D, E, F, G, H] = self.scope.get_data()
                data = dict(zip(keys, [x, A, B, C, D, E, F, G, H]))
                times, pulseheights = self._process_data(data)
                self._write_output(times, pulseheights)
                self._elapsed_time = time.time() - self._t_start_run
        self._close_output_file()
        self._write_info_file()
        return 0

    def acquire(self):

        self._set_channels()
        self._set_sampling()
        self._set_triggers()
        self.scope.set_up_buffers(self._num_samples, self._num_captures)
        if self._num_runs > 1:
            print(f'Acquiring {self._num_runs} runs.')
        for run in range(self._num_runs):
            self._acquire_run()

        return 0

def main():
    daq = udaq()
    daq.acquire()


if __name__ == '__main__':
    main()
