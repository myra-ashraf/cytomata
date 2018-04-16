import cv2
import pytest
import numpy as np
import matplotlib.pyplot as plt
from cytomata.interface import Microscope


def test_config_loaded():
    """Test that the Microscope object initialized correctly and loaded devices
    and presets from the micro-manager config file."""
    mic = Microscope()
    devices = mic.core.getLoadedDevices()
    groups = mic.core.getAvailableConfigGroups()
    assert devices
    assert groups


def test_change_channel():
    """Test that there is a 'Channel' group previously defined and we can
    correctly switch between some channels."""
    mic = Microscope()
    assert mic.core.isGroupDefined('Channel')
    mic.set_channel('mCherry')
    assert mic.core.getCurrentConfig('Channel') == 'mCherry'
    mic.set_channel('GFP')
    assert mic.core.getCurrentConfig('Channel') == 'GFP'


def test_capture_one_image():
    """Test that we can capture an image with the microscope camera and that
    it isn't blank or None."""
    mic = Microscope()
    assert mic.take_snapshot() is not None


def test_capture_consecutive_images():
    """Test that the image capture function can be called consecutively and
    still yield viable images."""
    mic = Microscope()
    assert mic.take_snapshot() is not None
    assert mic.take_snapshot() is not None


def test_fluorescence_determination(mic):
    """Test that the fluorescence intensity function yields a viable value."""
    mic = Microscope()
    img = mic.take_snapshot()
    roi_int, roi, bg_int, bg = mic.measure_fluorescence(img)
    assert roi is not None
    assert bg is not None
    assert roi_int
    assert bg_int
    assert roi_int > 0
    assert roi_int > bg_int
