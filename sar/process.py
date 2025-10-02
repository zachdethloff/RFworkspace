from sarpy.io.phase_history.converter import open_phase_history
import numpy as np
import matplotlib.pyplot as plt
import sarpy.processing

def sar_reader(size = 'full'):

    reader = open_phase_history('ICEYE_X34_CPHD_SLH_951662092_20251001T181929.cphd')

    print('reader type = {}'.format(type(reader)))  # see the explicit reader type

    print('image size = {}'.format(reader.data_size))


    signal_data = reader.read_signal_block()
    signal_data = signal_data['spot_0_burst_0']
    # Work with a tiny subset first
    if size == 'full':
        print(f"Image shape: {signal_data.shape}")
        return signal_data


    else:
        azimuth_subset = 10000  # Just 1000 azimuth lines
        range_subset = 500     # Just 500 range bins

        # Extract small chip from the middle
        az_start = (signal_data.shape[0] - azimuth_subset) // 2
        range_start = signal_data.shape[1] // 2

        chip = signal_data[az_start:az_start+azimuth_subset, 
                                range_start:range_start+range_subset]

        print(f"Large chip shape: {chip.shape}")
        print(f"Data size reduced from {signal_data.nbytes/1e9:.2f} GB to {chip.nbytes/1e6:.2f} MB")
        return chip


def main(type = 'basic'):
    image_data = sar_reader()
    if type == 'basic':
        # Use the FULL data without autofocus
        full_data = image_data  # shape: (81066, 26496)

        # Just do the azimuth FFT - that's the core of SAR processing
        image = np.fft.fftshift(np.fft.fft(full_data, axis=0), axes=0)

        # Convert to displayable form
        image_mag = np.abs(image)
        image_db = 20*np.log10(image_mag + 1e-10)

        # Display a subset (full image too large)
        plt.figure(figsize=(15, 10))
        subset = image_db[35000:40000, 10000:15000]  # 5000x5000 region
        plt.imshow(subset, cmap='gray', aspect='auto',
                vmin=np.percentile(subset, 2), 
                vmax=np.percentile(subset, 98))
        plt.colorbar(label='dB')
        plt.title('SAR Image - Basic FFT Processing')
        plt.xlabel('Range Bin')
        plt.ylabel('Azimuth Bin')
        plt.savefig('simple_process.png',dpi=300)
    else:
        chip = image_data
        range_power = np.mean(np.abs(chip)**2, axis=0)  # Average power per range bin

        # Plot to see where targets are
        plt.figure(figsize=(12, 4))
        plt.plot(range_power)
        plt.title('Average Power vs Range Bin')
        plt.xlabel('Range Bin')
        plt.ylabel('Power')
        plt.yscale('log')
        plt.savefig('avg_power.png',dpi=300,bbox_inches='tight')

        # Find the strongest target
        strongest_range_bin = np.argmax(range_power)
        print(f"Strongest return at range bin: {strongest_range_bin}")

        # Now look at the phase history for this strong target
        target_samples = chip[:, strongest_range_bin]
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
        plt.savefig('tmag.png',dpi=300,bbox_inches='tight')

        # Unwrap phase to see the true quadratic curve
        phase_unwrapped = np.unwrap(phase)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

        ax1.plot(magnitude)
        ax1.set_title(f'Magnitude - Range Bin {strongest_range_bin}')
        ax1.set_ylabel('Magnitude')

        ax2.plot(phase)
        ax2.set_title('Wrapped Phase')
        ax2.set_ylabel('Phase (radians)')

        ax3.plot(phase_unwrapped)
        ax3.set_title('Unwrapped Phase (should show quadratic curve)')
        ax3.set_ylabel('Phase (radians)')
        ax3.set_xlabel('Azimuth Sample')

        plt.tight_layout()
        plt.show()
        fig.savefig('plots.png',dpi=300,bbox_inches='tight')

        # Calculate the Doppler rate (linear phase slope)
        doppler_rate = np.polyfit(np.arange(len(phase_unwrapped)), phase_unwrapped, 1)[0]
        print(f"Doppler rate: {doppler_rate:.4f} radians/sample")
        print(f"Doppler frequency: {doppler_rate/(2*np.pi):.4f} cycles/sample")

        # Remove the linear trend to see if there's residual quadratic
        phase_detrended = phase_unwrapped - doppler_rate * np.arange(len(phase_unwrapped))

        plt.figure(figsize=(12, 4))
        plt.plot(phase_detrended)
        plt.title('Phase after removing linear trend (residual should show target defocus)')
        plt.ylabel('Phase (radians)')
        plt.xlabel('Azimuth Sample')
        plt.savefig('Phase_trend.png',dpi=300,bbox_inches='tight')


        # Fit a quadratic to the detrended phase
        n_samples = np.arange(len(phase_detrended))
        coeffs = np.polyfit(n_samples, phase_detrended, 2)
        a, b, c = coeffs  # ax² + bx + c

        print(f"Quadratic coefficient (a): {a:.6f}")
        print(f"Linear coefficient (b): {b:.6f}")  
        print(f"Constant (c): {c:.6f}")
        print(f"Peak location: {-b/(2*a):.1f} samples")

        # Plot the fit
        phase_fit = a*n_samples**2 + b*n_samples + c

        # Compare with and without autofocus
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        plt.figure(figsize=(12, 6))
        plt.plot(n_samples, phase_detrended, label='Actual phase', alpha=0.7)
        plt.plot(n_samples, phase_fit, 'r--', label='Quadratic fit', linewidth=2)
        plt.title('Quadratic Phase Fit')
        plt.xlabel('Azimuth Sample')
        plt.ylabel('Phase (radians)')
        plt.legend()
        plt.grid(True)
        plt.savefig('fit_and_detrend.png',dpi=300,bbox_inches='tight')

        # The azimuth matched filter is the conjugate of this phase
        Ka = -2 * a  # FM rate
        print(f"\nAzimuth FM rate (Ka): {Ka:.6f} radians/sample²")

        # CORRECTED FOCUSING PROCEDURE

        # Step 1: Remove the linear Doppler trend FIRST
        n_samples = np.arange(len(target_samples))
        linear_phase_correction = np.exp(-1j * doppler_rate * n_samples)
        detrended_samples = target_samples * linear_phase_correction

        # Step 2: Now apply quadratic matched filter to detrended data
        n_center = -b/(2*a)
        quadratic_phase = a * (n_samples - n_center)**2
        matched_filter = np.exp(-1j * quadratic_phase)  # Conjugate for compression

        focused_samples = detrended_samples * matched_filter

        # Step 3: FFT and compare
        unfocused_fft = np.fft.fftshift(np.fft.fft(target_samples))
        focused_fft = np.fft.fftshift(np.fft.fft(focused_samples))

        unfocused_mag = np.abs(unfocused_fft)
        focused_mag = np.abs(focused_fft)

        # Plot in dB scale for better visibility
        unfocused_db = 20*np.log10(unfocused_mag + 1e-10)
        focused_db = 20*np.log10(focused_mag + 1e-10)


        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(unfocused_db)
        ax1.set_title('Unfocused Target')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_ylim([np.max(unfocused_db)-40, np.max(unfocused_db)+5])
        ax1.grid(True)

        ax2.plot(focused_db)
        ax2.set_title('Focused Target (should show sharp peak at center)')
        ax2.set_ylabel('Magnitude (dB)')
        ax2.set_ylim([np.max(focused_db)-40, np.max(focused_db)+5])
        ax2.set_xlabel('Azimuth Bin')
        ax2.grid(True)

if __name__ == "__main__":
    main()