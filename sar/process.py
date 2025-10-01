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

range_power = np.mean(np.abs(small_chip)**2, axis=0)  # Average power per range bin

# Plot to see where targets are
plt.figure(figsize=(12, 4))
plt.plot(range_power)
plt.title('Average Power vs Range Bin')
plt.xlabel('Range Bin')
plt.ylabel('Power')
plt.yscale('log')
plt.show()

# Find the strongest target
strongest_range_bin = np.argmax(range_power)
print(f"Strongest return at range bin: {strongest_range_bin}")

# Now look at the phase history for this strong target
target_samples = small_chip[:, strongest_range_bin]
magnitude = np.abs(target_samples)
phase = np.angle(target_samples)

# Check if this range bin actually has a strong target
print(f"Target samples magnitude - min: {np.min(magnitude):.2f}, max: {np.max(magnitude):.2f}, mean: {np.mean(magnitude):.2f}")
print(f"Signal-to-noise estimate: {np.max(magnitude)/np.mean(magnitude):.2f}")

# Check the magnitude variation
plt.figure(figsize=(12, 4))
plt.plot(magnitude)
plt.title('Target Magnitude vs Azimuth Sample')
plt.ylabel('Magnitude')
plt.xlabel('Azimuth Sample')
plt.grid(True)
plt.show()

# If magnitude is relatively flat, this might not be a point target
# Point targets should show some magnitude modulation (the "envelope")

# Check what's in the metadata
print(reader.cphd_meta)

# Look for azimuth parameters
if hasattr(reader.cphd_meta, 'Channel'):
    print("\nChannel parameters:")
    print(reader.cphd_meta.Channel)

# Try to find FM rate or Doppler rate parameters

# Unwrap phase to see the true quadratic curve
# phase_unwrapped = np.unwrap(phase)

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

# ax1.plot(magnitude)
# ax1.set_title(f'Magnitude - Range Bin {strongest_range_bin}')
# ax1.set_ylabel('Magnitude')

# ax2.plot(phase)
# ax2.set_title('Wrapped Phase')
# ax2.set_ylabel('Phase (radians)')

# ax3.plot(phase_unwrapped)
# ax3.set_title('Unwrapped Phase (should show quadratic curve)')
# ax3.set_ylabel('Phase (radians)')
# ax3.set_xlabel('Azimuth Sample')

# plt.tight_layout()
# plt.show()
# fig.savefig('plots.png',dpi=300,bbox_inches='tight')

# # Calculate the Doppler rate (linear phase slope)
# doppler_rate = np.polyfit(np.arange(len(phase_unwrapped)), phase_unwrapped, 1)[0]
# print(f"Doppler rate: {doppler_rate:.4f} radians/sample")
# print(f"Doppler frequency: {doppler_rate/(2*np.pi):.4f} cycles/sample")

# # Remove the linear trend to see if there's residual quadratic
# phase_detrended = phase_unwrapped - doppler_rate * np.arange(len(phase_unwrapped))

# plt.figure(figsize=(12, 4))
# plt.plot(phase_detrended)
# plt.title('Phase after removing linear trend (residual should show target defocus)')
# plt.ylabel('Phase (radians)')
# plt.xlabel('Azimuth Sample')
# plt.savefig('Phase_trend.png',dpi=300,bbox_inches='tight')


# # Fit a quadratic to the detrended phase
# n_samples = np.arange(len(phase_detrended))
# coeffs = np.polyfit(n_samples, phase_detrended, 2)
# a, b, c = coeffs  # ax² + bx + c

# print(f"Quadratic coefficient (a): {a:.6f}")
# print(f"Linear coefficient (b): {b:.6f}")  
# print(f"Constant (c): {c:.6f}")
# print(f"Peak location: {-b/(2*a):.1f} samples")

# # Plot the fit
# phase_fit = a*n_samples**2 + b*n_samples + c

# plt.figure(figsize=(12, 6))
# plt.plot(n_samples, phase_detrended, label='Actual phase', alpha=0.7)
# plt.plot(n_samples, phase_fit, 'r--', label='Quadratic fit', linewidth=2)
# plt.title('Quadratic Phase Fit')
# plt.xlabel('Azimuth Sample')
# plt.ylabel('Phase (radians)')
# plt.legend()
# plt.grid(True)
# plt.savefig('fit_and_detrend.png',dpi=300,bbox_inches='tight')

# # The azimuth matched filter is the conjugate of this phase
# Ka = -2 * a  # FM rate
# print(f"\nAzimuth FM rate (Ka): {Ka:.6f} radians/sample²")

# # CORRECTED FOCUSING PROCEDURE

# # Step 1: Remove the linear Doppler trend FIRST
# n_samples = np.arange(len(target_samples))
# linear_phase_correction = np.exp(-1j * doppler_rate * n_samples)
# detrended_samples = target_samples * linear_phase_correction

# # Step 2: Now apply quadratic matched filter to detrended data
# n_center = -b/(2*a)
# quadratic_phase = a * (n_samples - n_center)**2
# matched_filter = np.exp(-1j * quadratic_phase)  # Conjugate for compression

# focused_samples = detrended_samples * matched_filter

# # Step 3: FFT and compare
# unfocused_fft = np.fft.fftshift(np.fft.fft(target_samples))
# focused_fft = np.fft.fftshift(np.fft.fft(focused_samples))

# unfocused_mag = np.abs(unfocused_fft)
# focused_mag = np.abs(focused_fft)

# # Plot in dB scale for better visibility
# unfocused_db = 20*np.log10(unfocused_mag + 1e-10)
# focused_db = 20*np.log10(focused_mag + 1e-10)

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# ax1.plot(unfocused_db)
# ax1.set_title('Unfocused Target')
# ax1.set_ylabel('Magnitude (dB)')
# ax1.set_ylim([np.max(unfocused_db)-40, np.max(unfocused_db)+5])
# ax1.grid(True)

# ax2.plot(focused_db)
# ax2.set_title('Focused Target (should show sharp peak at center)')
# ax2.set_ylabel('Magnitude (dB)')
# ax2.set_ylim([np.max(focused_db)-40, np.max(focused_db)+5])
# ax2.set_xlabel('Azimuth Bin')
# ax2.grid(True)
# fig.savefig('az_compress.png',dpi=300,bbox_inches='tight')

# # Measure peak-to-sidelobe ratio
# peak_focused = np.max(focused_mag)
# peak_bin = np.argmax(focused_mag)
# print(f"Peak at bin: {peak_bin} (should be near {len(focused_mag)//2})")
# print(f"Peak improvement: {20*np.log10(peak_focused/np.max(unfocused_mag)):.2f} dB")

# "LFMRate": -18011527377521.613,