import pytest
from cytomata.interface import Microscope
import matplotlib.pyplot as plt
import numpy as np
import cv2


def test_config_loaded(mic):
    """Test that the Microscope object initialized correctly and loaded devices
    and presets from the micro-manager config file."""
    devices = mic.core.getLoadedDevices()
    groups = mic.core.getAvailableConfigGroups()
    assert devices
    assert groups


def test_change_channel(mic):
    """Test that there is a 'Channel' group previously defined and we can
    correctly switch between some channels."""
    assert mic.core.isGroupDefined('Channel')
    mic.set_channel('mCherry')
    assert mic.core.getCurrentConfig('Channel') == 'mCherry'
    mic.set_channel('GFP')
    assert mic.core.getCurrentConfig('Channel') == 'GFP'


def test_capture_one_image(mic):
    """Test that we can capture an image with the microscope camera and that
    it isn't blank or None."""
    assert mic.take_snapshot() is not None


def test_capture_consecutive_images(mic):
    """Test that the image capture function can be called consecutively and
    still yield viable images."""
    assert mic.take_snapshot() is not None
    assert mic.take_snapshot() is not None


def test_fluorescence_determination(mic):
    """Test that the fluorescence intensity function yields a viable value."""
    img = mic.take_snapshot()
    roi_int, roi, bg_int, bg = mic.measure_fluorescence(img)
    assert roi is not None
    assert bg is not None
    assert roi_int
    assert bg_int
    assert roi_int > 0
    assert roi_int > bg_int
