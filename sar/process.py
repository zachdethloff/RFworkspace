from sarpy.io.phase_history.converter import open_phase_history
import numpy as np
import matplotlib.pyplot as plt
import sarpy.processing

reader = open_phase_history('ICEYE_X33_CPHD_SLF_951651468_20250927T164328.cphd')

print('reader type = {}'.format(type(reader)))  # see the explicit reader type

print('image size = {}'.format(reader.data_size))


signal_data = reader.read_signal_block()
signal_data = signal_data['spot_0_burst_0']
# Work with a tiny subset first
azimuth_subset = 1000  # Just 1000 azimuth lines
range_subset = 500     # Just 500 range bins

# Extract small chip from the middle
az_start = signal_data.shape[0] // 2
range_start = signal_data.shape[1] // 2

small_chip = signal_data[az_start:az_start+azimuth_subset, 
                        range_start:range_start+range_subset]

print(f"Small chip shape: {small_chip.shape}")
print(f"Data size reduced from {signal_data.nbytes/1e9:.2f} GB to {small_chip.nbytes/1e6:.2f} MB")

# Now analyze one range bin from this small chip
range_bin = 250  # Middle of our range subset
azimuth_samples = small_chip[:, range_bin]

# Plot the phase history
magnitude = np.abs(azimuth_samples)
phase = np.angle(azimuth_samples)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
ax1.plot(magnitude)
ax1.set_title('Magnitude vs Azimuth Sample (Small Subset)')
ax1.set_ylabel('Magnitude')

ax2.plot(phase)
ax2.set_title('Phase vs Azimuth Sample (Small Subset)')
ax2.set_ylabel('Phase (radians)')
ax2.set_xlabel('Azimuth Sample')
plt.tight_layout()
fig.savefig('plots.png',dpi=300,bbox_inches='tight')
# magnitude = np.abs(focused_im)
# image_mag = 20 * np.log10(magnitude + 1e-10)

# plt.figure(figsize=(12,8))
# plt.imshow(image_mag, cmap='gray',aspect='auto')
# plt.colorbar(label='dB')
# plt.title('Focused SAR Image')
# plt.savefig('sar_im.png',dpi=300, bbox_inches='tight')
# print('Figure Saved')