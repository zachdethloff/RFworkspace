import sarpy.io.phase_history.cphd as cphd_io
import numpy as np
import sarpy.io.general.tiff as tiff_io
from PIL import Image
import xml.etree.ElementTree as ET

# Read XML dimensions
tree = ET.parse('ICEYE_X11_CSI_SLED_2400501_20230722T131903.xml')
root = tree.getroot()
xml_azimuth = int(root.find('number_of_azimuth_samples').text)
xml_range = int(root.find('number_of_range_samples').text)
print(f"XML dimensions: {xml_azimuth} x {xml_range}")

# Check TIFF dimensions
img = Image.open('ICEYE_X11_CSI_SLED_2400501_20230722T131903.tif')
tiff_width, tiff_height = img.size
print(f"TIFF dimensions: {tiff_width} x {tiff_height}")
print(f"TIFF mode: {img.mode}")  # Should be complex or float for SAR
#from sarpy.processing.sicd_information import RangeCompression, AzimuthCompression


try:
    reader = tiff_io.TiffReader('ICEYE_X11_CSI_SLED_2400501_20230722T131903.tif')
    print(f"SARPy can read this TIFF: {reader is not None}")
    print(f"Data type: {reader.data_type}")
    if hasattr(reader, 'sicd_meta'):
        print("Contains SICD metadata")
except Exception as e:
    print(f"SARPy error: {e}")
