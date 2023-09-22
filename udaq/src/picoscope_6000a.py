"""Interface to a 6000 Series PicoScope.

Classes
-------
PicoScope6000A
    Interface to a 6000 Series PicoScope.

Basically taken from https://github.com/rileyle/udaq
based on 5000A scope and adjusted for 6000A
Many thanks to rileyle!!
"""


import ctypes
from threading import Event

import numpy as np

from picosdk.ps6000a import ps6000a as ps
from picosdk.PicoDeviceEnums import picoEnum as enums
from picosdk.PicoDeviceStructs import picoStruct as struct
from picosdk.functions import assert_pico_ok
from picosdk.constants import PICO_STATUS_LOOKUP

from itertools import combinations
from collections import defaultdict



INPUT_RANGES = {
    0.01: 1,
    0.02: 2,
    0.05: 3,
    0.1:  4,
    0.2:  5,
    0.5:  6,
    1:    7,
    2:    8,
    5:    9,
    10:   10,
    20:   11
}


class InvalidParameterError(Exception):
    """Error because of an invalid parameter."""
    pass


class PicoSDKError(Exception):
    """Error returned from PicoSDK."""
    pass


class DeviceNotFoundError(PicoSDKError):
    """Device cannot be found."""

    def __init__(self):
        super().__init__("Device not connected.")


class PicoScope6000A:
    """Interface to a 6000 Series PicoScope.

    This class encapsulates the low-level PicoSDK and offers a python-friendly
    interface to a 6000 Series PicoScope (e.g. a PicoScope 6XXXX).

    Methods
    -------
    open()
        open the device
    close()
        close the device
    set_channel()
        Set up input channels
    measure()
        Start a data collection run and return the data
    measure_adc_values()
        Start a data collection run and return the data in ADC values
    set_up_buffers()
        Set up memory buffers for reading data from device
    get_adc_data()
        Return all captured data, in ADC values
    get_data()
        Return all captured data, in physical units
    get_interval_from_timebase()
        Get sampling interval for given timebase
    start_run()
        Start a run in (rapid) block mode
    wait_for_data()
        Wait for device to finish data capture
    stop()
        Stop data capture
    set_trigger()
        Set the oscilloscope trigger condition
    set_advanced_triggers()
        Set advanced triggering

    """
    _handle = None

    def __init__(self, serial=None, resolution_bits='PICO_DR_8BIT'):
        """Instantiate the class and open the device."""
        self._channels_enabled = {'A': True, 'B': True, 'C': False, 'D': False, 'E': False, 'F': False, 'G': False, 'H': False}
        self._input_voltage_ranges = {}
        self._input_offsets = {}
        self._input_adc_ranges = {}
        self._buffers = {}
        self.data_is_ready = Event()
        self._callback = callback_factory(self.data_is_ready)
        self._resolution_bits = resolution_bits
        self.open(serial, resolution_bits)

    def __del__(self):
        """Instance destructor, close device."""
        if self._handle:
            self.stop()
            self.close()

    def open(self, serial=None, resolution_bits='PICO_DR_8BIT'):
        """Open the device.

        :param serial: (optional) Serial number of the device
        :param resolution_bits: vertical resolution in number of bits
        """
        handle = ctypes.c_int16()
        resolution = _get_resolution_from_bits(resolution_bits)
        status = ps.ps6000aOpenUnit(ctypes.byref(handle), serial, resolution)
        status_msg = PICO_STATUS_LOOKUP[status]

        if status_msg == "PICO_OK":
            self._handle = handle
        elif status_msg == 'PICO_NOT_FOUND':
            raise DeviceNotFoundError()
        else:
            raise PicoSDKError(f"PicoSDK returned {status_msg}")

        self.set_channel('A', is_enabled=False)
        self.set_channel('B', is_enabled=False)
        self.set_channel('C', is_enabled=False)
        self.set_channel('D', is_enabled=False)
        self.set_channel('E', is_enabled=False)
        self.set_channel('F', is_enabled=False)
        self.set_channel('G', is_enabled=False)
        self.set_channel('H', is_enabled=False)

    def close(self):
        """Close the device."""
        assert_pico_ok(ps.ps6000aCloseUnit(self._handle))
        self._handle = None

    def set_channel(self, channel_name, coupling_type='DC', range_value=1,
                    offset=0, is_enabled=True):
        """Set up input channels.

        :param channel_name: channel name ('A', 'B', etc.)
        :param coupling_type: 'AC' or 'DC' coupling
        :param range_value: (float) input voltage range in volts
        :param offset: analogue offset of the input signal
        :param is_enabled: enable or disable the channel
        :type is_enabled: boolean

        The input voltage range can be 10, 20, 50 mV, 100, 200, 500 mV, 1, 2,
        5 V or 10, 20, 50 V, but is given in volts. For example, a range of
        20 mV is given as 0.02.
        """
        channel = _get_channel_from_name(channel_name)
        coupling_type = _get_coupling_type_from_name(coupling_type)
        range_val = _get_range_from_value(range_value) 
        bandwidth = enums.PICO_BANDWIDTH_LIMITER["PICO_BW_FULL"]
        if is_enabled:
            assert_pico_ok(ps.ps6000aSetChannelOn(self._handle, channel,
                                                  coupling_type, range_val, offset, bandwidth))
        else:
            assert_pico_ok(ps.ps6000aSetChannelOff(self._handle, channel))


        self._input_voltage_ranges[channel_name] = float(range_value)
        self._input_offsets[channel_name] = float(offset)
        max_adc_value = ctypes.c_int16()
        min_adc_value = ctypes.c_int16()
        resolution    = enums.PICO_DEVICE_RESOLUTION[self._resolution_bits]
        assert_pico_ok(ps.ps6000aGetAdcLimits(self._handle, resolution, ctypes.byref(min_adc_value), ctypes.byref(max_adc_value)))
        self._input_adc_ranges[channel_name] = max_adc_value.value
        self._channels_enabled[channel_name] = is_enabled

    def measure(self, num_pre_samples, num_post_samples, timebase=4,
                num_captures=1):
        """Start a data collection run and return the data.

        Start a data collection run and collect a number of captures. The data
        is returned as a twodimensional NumPy array (i.e. a 'list' of captures)
        with data values in volts. An array of time values in seconds is also
        returned.

        :param num_pre_samples: number of samples before the trigger
        :param num_post_samples: number of samples after the trigger
        :param timebase: timebase setting (see programmers guide for reference)
        :param num_captures: number of captures to take

        :returns: time_values, data
        """
        data = self.measure_adc_values(num_pre_samples, num_post_samples,
                                       timebase, num_captures)

        num_samples = num_pre_samples + num_post_samples
        time_values = self._calculate_time_values(timebase, num_samples)

        V_data = []
        for channel, values in zip(self._channels_enabled, data):
            if self._channels_enabled[channel] is True:
                V_data.append(self._rescale_adc_to_V(channel,
                                                     np.array(values)))
            else:
                V_data.append(None)

        return time_values, V_data

    def measure_adc_values(self, num_pre_samples, num_post_samples, timebase=4,
                           num_captures=1):
        """Start a data collection run and return the data in ADC values.

        Start a data collection run in 'rapid block mode' and collect a number
        of captures. The data is returned as a list of NumPy arrays (i.e. a
        list of captures) with unconverted ADC values.

        :param num_pre_samples: number of samples before the trigger
        :param timebase: timebase setting (see programmers guide for reference)
        :param num_captures: number of captures to take

        :returns: data
        """
        num_samples = num_pre_samples + num_post_samples
        self.set_up_buffers(num_samples, num_captures)
        self.start_run(num_pre_samples, num_post_samples, timebase,
                       num_captures)
        self.wait_for_data()
        values = self._get_values(num_samples, num_captures)

        self.stop()
        return values

    def set_up_buffers(self, num_samples, num_captures=1):
        """Set up memory buffers for reading data from device.

        :param num_samples: the number of required samples per capture.
        :param num_captures: the number of captures.
        """
        self._set_memory_segments(num_captures, num_samples)
        for ic, channel in enumerate(self._get_enabled_channels()):
            if ic == 0:
                do_clear = True
            self._set_data_buffer(channel, num_samples, num_captures, do_clear)
            do_clear = False

    def get_adc_data(self):
        """Return all captured data, in ADC values."""
        return self._get_values(self._num_samples, self._num_captures)

    def get_data(self):
        """Return all captured data, in physical units.

        This method returns a tuple of time values (in seconds) and the
        captured data (in Volts).

        """
        data = self.get_adc_data()
        if data is None:
            return None, [None, None, None, None, None, None, None, None]
        time_values = self._calculate_time_values(self._timebase,
                                                  self._num_samples)

        V_data = []
        for channel, values in zip(self._channels_enabled, data):
            if self._channels_enabled[channel] is True:
                V_data.append(self._rescale_adc_to_V(channel,
                                                     np.array(values)))
            else:
                V_data.append(None)

        return time_values, V_data

    def _calculate_time_values(self, timebase, num_samples):
        """Calculate time values from timebase and number of samples."""
        interval = self.get_interval_from_timebase(timebase, num_samples)
        return interval * np.arange(num_samples) * 1e-9

    def _rescale_adc_to_V(self, channel, data):
        """Rescale the ADC data and return float values in volts.

        :param channel: name of the channel
        :param data: the data to transform
        """
        voltage_range = self._input_voltage_ranges[channel]
        offset = self._input_offsets[channel]
        max_adc_value = self._input_adc_ranges[channel]
        return (voltage_range * data) / max_adc_value - offset

    def _rescale_V_to_adc(self, channel, data):
        """Rescale float values in volts to ADC values.

        :param channel: name of the channel
        :param data: the data to transform
        """
        voltage_range = self._input_voltage_ranges[channel]
        offset = self._input_offsets[channel]
        max_adc_value = self._input_adc_ranges[channel]
        output = max_adc_value * (data + offset) / voltage_range
        try:
            return output.astype(np.int16)
        except AttributeError:
            return int(output)

    def get_interval_from_timebase(self, timebase, num_samples=100):
        """Get sampling interval for given timebase.

        :param timebase: timebase setting (see programmers guide for reference)
        :param num_samples: number of samples required

        :returns: sampling interval in nanoseconds
        """
        interval = ctypes.c_double() 
        timebase = ctypes.c_int32(timebase) 
        enabledChannelFlags = 0
        for ch in 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H':           
            enabledChannelFlags += enums.PICO_CHANNEL_FLAGS[f"PICO_CHANNEL_{ch}_FLAGS"]                               
        assert_pico_ok(ps.ps6000aGetTimebase(
            self._handle, timebase.value, num_samples, ctypes.byref(interval), None,
            0))
        
        return interval.value

    def _set_memory_segments(self, num_segments, num_samples):
        """Set up memory segments in the device.

        For multiple captures, the device's memory must be divided into
        segments, one for each capture. If the memory cannot contain the
        required number of samples per segment, an InvalidParameterError is
        raised.

        :param num_segments: the number of memory segments.
        :param num_samples: the number of required samples per capture.
        """
        max_samples = ctypes.c_int32()
        print("Number of segments:",num_segments)
        assert_pico_ok(ps.ps6000aMemorySegments(self._handle, num_segments,
                                                ctypes.byref(max_samples)))
        max_samples = max_samples.value
        print("Maximum number of segments possible:",max_samples)
        if max_samples < num_samples:
            raise InvalidParameterError(
                f"A memory segment can only fit {max_samples}, but "
                f"{num_samples} are required.")

    def _set_data_buffer(self, channel_name, num_samples, num_captures=1, do_clear=False):
        """Set up data buffer.

        :param channel_name: channel name ('A', 'B', etc.)
        :param num_samples: number of samples required
        :param num_captures: number of captures
        """
        down_sample_mode = enums.PICO_RATIO_MODE["PICO_RATIO_MODE_RAW"]
        data_type = enums.PICO_DATA_TYPE["PICO_INT16_T"]
        clear = enums.PICO_ACTION["PICO_CLEAR_ALL"]
        add = enums.PICO_ACTION["PICO_ADD"]
        action = add 
        if do_clear is True:
            action = clear+add
        channel = _get_channel_from_name(channel_name)
        self._buffers[channel_name] = [ (ctypes.c_int16*num_samples)() for i in range(num_captures)]
        for segment in range(num_captures):
            assert_pico_ok(ps.ps6000aSetDataBuffer(
                self._handle, channel, ctypes.byref(self._buffers[channel_name][segment]), 
                num_samples, data_type, segment, down_sample_mode, action))
            action=add
 
    def start_run(self, num_pre_samples, num_post_samples, timebase=4,
                  num_captures=1, callback=None):
        """Start a run in (rapid) block mode.

        Start a data collection run in 'rapid block mode' and collect a number
        of captures. Unlike the :method:`measure` and
        :method:`measure_adc_values`, which handle all details for you, this
        method returns immediately, while the device captures the requested
        data. Make sure that you *first* run :method:`set_up_buffers` to set up
        the device's memory buffers. You can supply a C-style callback to be
        notified when the device is ready, or call :method:`wait_for_data:.
        When the data is ready, call :method:`get_data` or
        :method:`get_adc_data`. When done measuring data, make sure to call
        :method:`stop`.

        :param num_pre_samples: number of samples before the trigger
        :param num_post_samples: number of samples after the trigger
        :param timebase: timebase setting (see programmers guide for reference)
        :param num_captures: number of captures to take

        :returns: data
        """
        # save samples and captures for reference
        self._num_samples = num_pre_samples + num_post_samples
        self._timebase = timebase
        self._num_captures = num_captures

        # # if callback is None:
        # #     callback = self._callback
        # # self.data_is_ready.clear()
        
        time_indisposed_ms = ctypes.c_double(0)
        assert_pico_ok(ps.ps6000aSetNoOfCaptures(self._handle, num_captures))
        assert_pico_ok(ps.ps6000aRunBlock(self._handle, num_pre_samples, num_post_samples, timebase, ctypes.byref(time_indisposed_ms), 0, None, None))
        

    def wait_for_data(self):
        """Wait for device to finish data capture."""
        ##self.data_is_ready.wait()

        # Check for data collection to finish using ps6000aIsReady
        ready = ctypes.c_int16(0)
        assert_pico_ok(ps.ps6000aIsReady(self._handle, ctypes.byref(ready)))
        return ready.value

    def _get_values(self, num_samples, num_captures):
              
        """Get data from device and return buffer or None."""
        num_samples = ctypes.c_uint32(num_samples)
        overflow = (ctypes.c_int16 * num_captures)()
        down_sample_mode = enums.PICO_RATIO_MODE["PICO_RATIO_MODE_RAW"]
        status = ps.ps6000aGetValuesBulk(
                self._handle, 0, ctypes.byref(num_samples), 0, num_captures - 1, 1, down_sample_mode, ctypes.byref(overflow))
       
        
        status_msg = PICO_STATUS_LOOKUP[status]
        if status_msg == "PICO_OK":
            return [self._buffers[channel] if is_enabled is True else None
                    for channel, is_enabled in self._channels_enabled.items()]
        elif status_msg == "PICO_NO_SAMPLES_AVAILABLE":
            return None
        else:
            raise PicoSDKError(f"PicoSDK returned {status_msg}")

    def stop(self):
        """Stop data capture."""
        assert_pico_ok(ps.ps6000aStop(self._handle))

    def set_trigger(self, channel_name, threshold=0., direction='RISING',
                    is_enabled=True, delay=0, auto_trigger=0):
        """Set the oscilloscope trigger condition.

        :param channel_name: the source channel for the trigger (e.g. 'A')
        :param threshold: the trigger threshold (in V)
        :param direction: the direction in which the signal must move to cause
            a trigger
        :param is_enabled: (boolean) enable or disable the trigger
        :param delay: the delay between the trigger occuring and the start of
            capturing data, in number of sample periods
        :param auto_trigger: the maximum amount of time to wait for a trigger
            before starting capturing data in seconds

        The direction parameter can take values of 'ABOVE', 'BELOW', 'RISING',
        'FALLING' or 'RISING_OR_FALLING'.
        """
        channel = _get_channel_from_name(channel_name)
        threshold = self._rescale_V_to_adc(channel_name, threshold)
        direction = _get_trigger_direction_from_name(direction)
        assert_pico_ok(ps.ps5000aSetSimpleTrigger(
            self._handle, is_enabled, channel, threshold, direction, delay,
            auto_trigger))

    def set_advanced_triggers(self, is_enabled, type, direction, threshold, combo_type=2):
        """Set advanced triggering. Individual channel conditions are combined
           in a logical OR.

        :param is_enabled (list): boolean: enable or disable for each channel
        :param type (list): LEVEL or WINDOW (only LEVEL supported for now)
        :param direction (list): the direction in which the signal must move
            to cause a trigger for each channel
        :param threshold (list): trigger threshold (in V) for each channel

        The direction parameter can take values of 'ABOVE', 'BELOW', 'RISING',
        'FALLING' or 'RISING_OR_FALLING'.
        """
        
        trigChanEnabledList = []
        num_enabled = sum(list(is_enabled.values()))
        trigDirList = []
        trigPropList = []
        for ch in 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H':
             if is_enabled[ch]:
                channel = _get_channel_from_name(ch)
                trigChanEnabledList.append(ch)
                trig_dir = _get_trigger_direction_from_name(direction[ch])
                trig_type = enums.PICO_THRESHOLD_MODE["PICO_LEVEL"]
                trigDirList.append(struct.PICO_DIRECTION(channel, trig_dir, trig_type))
                thresh = self._rescale_V_to_adc(ch, threshold[ch])
                prop = struct.PICO_TRIGGER_CHANNEL_PROPERTIES(thresh,0, 0, 0, channel)
                trigPropList.append(prop)

        Directions = struct.PICO_DIRECTION*num_enabled
        trigDirections = Directions(*trigDirList)

        Properties = struct.PICO_TRIGGER_CHANNEL_PROPERTIES*num_enabled
        trigProperties = Properties(*trigPropList)

        assert_pico_ok(ps.ps6000aSetTriggerChannelDirections(self._handle,
            ctypes.byref(trigDirections), num_enabled))
        
        autoTriggerMicroSeconds = 0
        assert_pico_ok(ps.ps6000aSetTriggerChannelProperties(self._handle,
            ctypes.byref(trigProperties), num_enabled, 0, autoTriggerMicroSeconds))

        # initialise trigger logic flags
        clear      = enums.PICO_ACTION["PICO_CLEAR_ALL"]
        add        = enums.PICO_ACTION["PICO_ADD"]
        flag = clear 
        state_true = enums.PICO_TRIGGER_STATE["PICO_CONDITION_TRUE"]
        
        # pick unique singles, doubles, triples...
        combos = list(combinations(trigChanEnabledList,r=combo_type))
        print("These are the trigger combinations that will be used \n", combos)
       
        for icombo, combo in enumerate(combos):
            num_used = len(combo)
            print("Processing trigger combination:", combo)
            trigger_conditions = (struct.PICO_CONDITION * num_used)()
            for ich, ch in enumerate(combo):
                trigger_conditions[ich] = struct.PICO_CONDITION(_get_channel_from_name(ch), 
                                                                enums.PICO_TRIGGER_STATE["PICO_CONDITION_TRUE"])
            n_trig_cond = num_used    
            assert_pico_ok(ps.ps6000aSetTriggerChannelConditions(self._handle, ctypes.byref(trigger_conditions), 
                                                                 n_trig_cond, flag))
            print("The flag is:",flag)
            flag = add

    def _get_enabled_channels(self):
        """Return list of enabled channels."""
        return [channel for channel, status in self._channels_enabled.items()
                if status is True]


def _get_resolution_from_bits(resolution_bits):
    """Return the resolution from the number of bits."""
    if resolution_bits in ['PICO_DR_12BIT','PICO_DR_10BIT','PICO_DR_8BIT']:
        def_name = resolution_bits
    else:
        raise InvalidParameterError(f"A resolution of {def_name}-bits "
                                    "is not supported")
    return enums.PICO_DEVICE_RESOLUTION[def_name]


def _get_channel_from_name(channel_name):
    """Return the channel from the channel name."""
    if channel_name in ['A', 'B', 'C', 'D','E', 'F', 'G', 'H']:
        def_name = f"PICO_CHANNEL_{channel_name}"
    else:
        raise InvalidParameterError(f"Channel {channel_name} is not supported")
    return enums.PICO_CHANNEL[def_name]


def _get_coupling_type_from_name(coupling_type_name):
    """Return the coupling type from the coupling type name."""
    if coupling_type_name in ['AC', 'DC', 'DC_50OHM']:
        def_name = f"PICO_{coupling_type_name}"
    else:
        raise InvalidParameterError(f"Coupling type {coupling_type_name} is "
                                    "not supported")
    return enums.PICO_COUPLING[def_name]


def _get_range_from_value(range):
    """Return the range from the range in volts."""
    range_val = -1
    if range in INPUT_RANGES:
        range_val = INPUT_RANGES[range]
        def_name = f"PICO_{range_val}"
    else:
        raise InvalidParameterError(f"Range {range} V is not supported")
    return range_val 


def _get_trigger_direction_from_name(direction_name):
    """Return the trigger direction from the direction name."""
    if direction_name in ['ABOVE', 'BELOW', 'RISING', 'FALLING',
                          'RISING_OR_FALLING']:
        def_name = f"PICO_{direction_name}"
    else:
        raise InvalidParameterError(f"Trigger direction {direction_name} is "
                                    "not supported")
    return enums.PICO_THRESHOLD_DIRECTION[def_name]


def callback_factory(event):
    """Return callback that will signal event when called."""
    @ctypes.CFUNCTYPE(None, ctypes.c_int16, ctypes.c_int, ctypes.c_void_p)
    def data_is_ready_callback(handle, status, parameters):
        """Signal that data is ready when called by PicoSDK."""
        event.set()
    return data_is_ready_callback
