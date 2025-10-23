from sarpy.io.phase_history.converter import open_phase_history
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sarpy.processing
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count

def sar_reader(
        cphd
):

    # range_to_pixel = np.load(rtp)
    reader = open_phase_history(cphd)
    data = reader.cphd_meta.to_dict()
    geo_lib = data['ReferenceGeometry']
    scene_surface = data['SceneCoordinates']['ReferenceSurface']['Planar']
    chan_lib = data['Channel']['Parameters']
    global_lib = data['Global']['TOASwath']
    coord = ['X','Y','Z']
    Rcv = data['TxRcv']['RcvParameters']

    
    params = {
        'center_freq' : chan_lib[0]['FxC'],
        'scp' : np.array([geo_lib['SRP']['ECF'][val] for val in coord]),
        'collection_duration' : geo_lib['SRPDwellTime'],
        'scp_time' : geo_lib['ReferenceTime'],
        'row_uvect' : np.array([scene_surface['uIAX'][val] for val in coord]),
        'col_uvect' : np.array([scene_surface['uIAY'][val] for val in coord]),
        'sample_rate' : Rcv[0]['SampleRate']

    }

    print('reader type = {}'.format(type(reader)))  # see the explicit reader type

    print('image size = {}\n'.format(reader.data_size))


    signal_data = reader.read_signal_block()
    signal_data = signal_data['spot_0_burst_0']

    return signal_data, params, reader

def pulse_analysis(cphd_data,reader,test):
    if test == 'single':
        print('Starting single pulse')
        test_data = cphd_data[10000, :]

        toa2 = reader.read_pvp_variable('TOA2', index=0, the_range=None)
        t_spacing = np.linspace(0, toa2, len(test_data))
        t_us = t_spacing * 1e6
        powert = 20*np.log10(np.abs(test_data)+1e-10)
        print(len(test_data),len(powert))
        print('Power Converted')

        plt.figure(figsize=(12, 5))
        plt.plot(t_us[::100], powert[::100])
                #aspect='auto', cmap='gray', vmin=-40, vmax=0)
        plt.xlabel('Time us')
        plt.ylabel('Power (dB)')
        plt.title('Single Pulse Time Domain')
        #plt.colorbar(label='Magnitude (dB)')
        plt.savefig('t_domain_pulse.png',dpi=300,bbox_inches='tight')

        # FFT in azimuth direction (along pulses)
        fft_result = np.fft.fftshift(np.fft.fft(test_data, axis=0), axes=0)

        freqs = np.fft.fftshift(np.fft.fftfreq(len(test_data), 1/params['sample_rate']))
        freqs_mhz = freqs / 1e6

        power = 20*np.log10(np.abs(fft_result)+1e-10)
        mask = freqs_mhz > 70

        plt.figure(figsize=(12, 8))
        plt.plot(freqs_mhz[::100], power[::100])
                #aspect='auto', cmap='gray', vmin=-40, vmax=0)
        plt.xlabel('Frequency MHz')
        plt.ylabel('Power (dB)')
        plt.title('Single Pulse Frequency Domain')
        #plt.colorbar(label='Magnitude (dB)')
        plt.savefig('f_domain_pulse.png',dpi=300,bbox_inches='tight')

        filtered_ph = fft_result.copy()
        filtered_ph[~mask] = 0

        new_ph = np.fft.ifftshift(filtered_ph)
        new_ph = np.fft.ifft(new_ph)

        power_filtered = 20*np.log10(np.abs(new_ph)+1e-10)
        plt.figure(figsize=(12, 5))
        plt.plot(t_us[::100], power_filtered[::100])
                #aspect='auto', cmap='gray', vmin=-40, vmax=0)
        plt.xlabel('Time us')
        plt.ylabel('Power (dB)')
        plt.title('Single Pulse Time Domain')
        #plt.colorbar(label='Magnitude (dB)')
        plt.savefig('t_domain_pulse_filtered.png',dpi=300,bbox_inches='tight')

        power_filtered = 20*np.log10(np.abs(filtered_ph)+1e-10)

        plt.figure(figsize=(12, 8))
        plt.plot(freqs_mhz[::100], power_filtered[::100])
                #aspect='auto', cmap='gray', vmin=-40, vmax=0)
        plt.xlabel('Frequency MHz')
        plt.ylabel('Power (dB)')
        plt.title('Single Pulse Frequency Domain')
        #plt.colorbar(label='Magnitude (dB)')
        plt.savefig('f_domain_pulse_filtered.png',dpi=300,bbox_inches='tight')

        print('Single Pulse complete')
    else:
        test_data = cphd_data[:1000, 100]

        fft_out = np.fft.fftshift(np.fft.fft(test_data,axis=0),axes=0)

    

class SARBackprojection:
    def __init__(self, cphd_data, params, reader):
        """
        Initialize SAR Backprojection processor
        
        Args:
            cphd_data: Complex phase history data (num_pulses, num_range_bins)
            range_bins: Range values for each bin (meters)
            sicd_params: Dictionary with SICD parameters
        """
        self.cphd = cphd_data
        self.num_pulses, self.num_range_bins = cphd_data.shape

        print(self.num_pulses)

        # From SICD
        row_ss = 0.58445480567920705  # Sample spacing in meters

        # Starting range of image
        range_start = range_bins[0]

        # Generate full range array using SICD data instead of interpolation
        full_range_bins = range_start + np.arange(self.num_range_bins) * row_ss

        if len(full_range_bins) == self.num_range_bins:
            self.range_bins = full_range_bins
            print("Interpolated Ranges")
        
        # Extract SICD parameters
        self.sicd_params = sicd_params
        self.arp_poly_x = sicd_params['arp_poly_x'].astype(float) # ARP X Polynomial
        self.arp_poly_y = sicd_params['arp_poly_y'].astype(float) # ARP Y Polynomial
        self.arp_poly_z = sicd_params['arp_poly_z'].astype(float) # ARP Z Polynomial
        self.center_freq = sicd_params['center_freq'] # Frequency Center
        self.wavelength = 3e8 / self.center_freq
        self.scp = sicd_params['scp']  # Scene center point
        self.collect_start = sicd_params['collection_start'] 
        self.collect_duration = sicd_params['collection_duration']
        self.scp_time = sicd_params['scp_time'] # Time at SCP
        self.row_uvect = sicd_params['row_uvect'] # Pixel range spacing
        self.col_uvect = sicd_params['col_uvect'] # Pixel azimuth spacing
        self.arp_pos = sicd_params['arp_pos_scp'] # Position per pulse
        self.arp_vel = sicd_params['arp_vel_scp'] # Velocity per pulse
        self.arp_acc = sicd_params['arp_acc_scp'] # Acceleration per pulse
        
        # Compute time for each pulse
        # self.pulse_times = np.linspace(0, self.collect_duration, self.num_pulses)

        pulse_indices = np.arange(self.num_pulses)
        self.pulse_times = pulse_indices / 6261.785
        print(f"Pulse times: {self.pulse_times[0]:.6f} to {self.pulse_times[-1]:.6f} seconds")
        print(f"Should span: 0 to 4.521 seconds")
        
        # Precompute sensor positions for all pulses
        self.sensor_positions = self._compute_sensor_positions()
        self.diagnostic_checks('Mag')

    def diagnostic_checks(self,check):
        if check == 'Mag':
            # Look at the raw CPHD to find strong returns
            max_per_pulse = np.max(np.abs(self.cphd), axis=1)
            strongest_pulse = np.argmax(max_per_pulse)
            strongest_range_bin = np.argmax(np.abs(self.cphd[strongest_pulse, :]))

            print(f"Magnitude of Strongest Pulse: {np.abs(self.cphd[strongest_pulse, strongest_range_bin]):.1f}")

            target_range = self.range_bins[strongest_range_bin]  # Should work now!
            print(f"Strongest scatterer at range: {target_range:.1f} m")
        else:

            print(f"Full range bins: {self.range_bins[0]:.1f} to {self.range_bins[-1]:.1f} m")
            print(f"Range span: {self.range_bins[-1] - self.range_bins[0]:.1f} m")
            print(f"Number of bins: {len(self.range_bins)}")

            # Verify against your geometry
            print(f"\nGeometry check:")
            print(f"Expected slant range at SCP: 560739 m")
            print(f"Range bins cover: {self.range_bins[0]:.1f} to {self.range_bins[-1]:.1f} m")

            scp_time = 2.2606971263885498
            t_rel = scp_time - self.scp_time  # Should be 0!
            print('Relative T: ', t_rel)

            pos_x = np.polyval(self.arp_poly_x[::-1], t_rel)
            pos_y = np.polyval(self.arp_poly_y[::-1], t_rel)
            pos_z = np.polyval(self.arp_poly_z[::-1], t_rel)

            print(f"ARPPoly at t=0: [{pos_x}, {pos_y}, {pos_z}]")
            print(f"SCPCOA ARPPos:  [2477545.11, 4892263.42, 4062226.54]")
        
    def _compute_sensor_positions(self):
        """Compute sensor position for each pulse using ARPPoly"""
        positions = np.zeros((self.num_pulses, 3)) # Position matrix for each pulse

        if 'arp_pos_scp' in self.sicd_params.keys():
            print("Taylor Approximation Method Selected")
            
            # Taylor Series approximation of each pulse
            for i, t in enumerate(self.pulse_times):
                dt = t - self.scp_time
                positions[i] = self.arp_pos + self.arp_vel * dt + 0.5 * self.arp_acc * dt**2
        else:

            
            for i, t in enumerate(self.pulse_times):
                # Evaluate polynomial manually
                relt = t - self.scp_time
                positions[i, 0] = np.polyval(self.arp_poly_x[::-1], relt)  # X
                positions[i, 1] = np.polyval(self.arp_poly_y[::-1], relt)  # Y
                positions[i, 2] = np.polyval(self.arp_poly_z[::-1], relt)  # Z
                
        return positions
    
    def create_image_grid(self, image_size_m=1000, pixel_spacing_m=1.0):
        """
        Create image grid centered on SCP
        
        Args:
            image_size_m: Size of image in meters (both dimensions)
            pixel_spacing_m: Spacing between pixels in meters
        
        Returns:
            X, Y, Z: 2D meshgrids of pixel positions in ECEF coordinates
        """
        n_pixels = int(image_size_m / pixel_spacing_m)
        
        # Local coordinate offsets
        row_offsets = np.linspace(-image_size_m/2, image_size_m/2, n_pixels)
        col_offsets = np.linspace(-image_size_m/2, image_size_m/2, n_pixels)
        Col_grid, Row_grid = np.meshgrid(col_offsets, row_offsets)
        
        # Get unit vectors from SICD (if available)
        row_uvect = self.row_uvect
        col_uvect = self.col_uvect
        
        if row_uvect is None or col_uvect is None:
            # Fallback: compute from geometry 
            avg_sensor_pos = np.mean(self.sensor_positions, axis=0)
            look_vec = avg_sensor_pos - self.scp
            look_vec = look_vec / np.linalg.norm(look_vec)
            ground_normal = self.scp / np.linalg.norm(self.scp)
            range_dir = np.cross(ground_normal, look_vec)
            range_dir = range_dir / np.linalg.norm(range_dir)
            az_dir = np.cross(range_dir, ground_normal)
            az_dir = az_dir / np.linalg.norm(az_dir)
            row_uvect = range_dir
            col_uvect = az_dir
        
        # Build 3D grid using SICD vectors: Starting position + Row Coordinate * Row Vector + Column Coordinate * Column Vector
        X_ecef = self.scp[0] + Row_grid * row_uvect[0] + Col_grid * col_uvect[0]
        Y_ecef = self.scp[1] + Row_grid * row_uvect[1] + Col_grid * col_uvect[1]
        Z_ecef = self.scp[2] + Row_grid * row_uvect[2] + Col_grid * col_uvect[2]
        
        return X_ecef, Y_ecef, Z_ecef
    
    def backproject(self, image_grid, pulse_subset=None, n_workers=4, use_process=False):
        """
        Perform backprojection on image grid
        
        Args:
            image_grid: Tuple of (X, Y, Z) meshgrids in ECEF coordinates
            pulse_subset: Optional indices of pulses to use (for faster processing)
        
        Returns:
            Complex image (n_y, n_x)
        """
        X_grid, Y_grid, Z_grid = image_grid
        image = np.zeros_like(X_grid, dtype=complex)
            
        # Use all pulses or subset
        if pulse_subset is None:
            pulse_indices = range(self.num_pulses)
        else:
            pulse_indices = pulse_subset

        if n_workers is None:
            n_workers = cpu_count()
        
        print(f"Backprojecting {len(pulse_indices)} pulses with {n_workers} workers...")
        print(f"Method is {'multiprocessing' if use_process else 'multithreading'}")

        chunk_size = max(1, len(pulse_indices) // (n_workers * 4))
        pulse_chunks = [pulse_indices[i:i+chunk_size] 
                       for i in range(0, len(pulse_indices), chunk_size)]
        
        # Choose executor type
        ExecutorClass = ProcessPoolExecutor if use_process else ThreadPoolExecutor
        
        # Process chunks in parallel
        with ExecutorClass(max_workers=n_workers) as executor:
            futures = [executor.submit(self.process_pulse_chunk, 
                                      chunk, X_grid, Y_grid, Z_grid) 
                      for chunk in pulse_chunks]
            
            # Collect results and accumulate
            for i, future in enumerate(futures):
                chunk_image = future.result()
                image += chunk_image
                if (i + 1) % max(1, len(futures) // 10) == 0:
                    print(f"  Completed {i+1}/{len(futures)} chunks")

        # 1. Check if you're getting any coherent signal
        print("Image statistics:")
        print(f"Mean magnitude: {np.mean(np.abs(image))}")
        print(f"Max magnitude: {np.max(np.abs(image))}")
        print(f"Std magnitude: {np.std(np.abs(image))}")

        return image

    def process_pulse_chunk(self, pulse_indices, X_grid, Y_grid, Z_grid):

        partial_image = np.zeros_like(X_grid, dtype=complex)
        
        # Main backprojection loop
        for pulse_idx in pulse_indices:
            if pulse_idx % 1000 == 0:
                print(f"  Processing pulse {pulse_idx}/{self.num_pulses}")
            
            # Get sensor position for this pulse
            sensor_pos = self.sensor_positions[pulse_idx]
            
            # Compute range from sensor to each pixel
            dx = X_grid - sensor_pos[0]
            dy = Y_grid - sensor_pos[1]
            dz = Z_grid - sensor_pos[2]
            ranges = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Find corresponding range bin for each pixel
            # Use linear interpolation
            range_indices = np.interp(ranges.ravel(), self.range_bins, 
                                     np.arange(len(self.range_bins)))
            range_indices = range_indices.reshape(ranges.shape)
            

            # Get complex values from phase history
            # Clip indices to valid range
            valid_mask = (range_indices >= 0) & (range_indices < self.num_range_bins - 1)

            idx_low = np.floor(range_indices).astype(int)
            idx_high = np.minimum(np.ceil(range_indices).astype(int), range_indices - 1).astype(int)
            alpha = range_indices - idx_low

            low_interp = self.cphd[pulse_idx, idx_low]
            high_interp = self.cphd[pulse_idx, idx_high]
            val = (1 - alpha)*low_interp + alpha*high_interp

            val = val * valid_mask

            phase_correction = np.exp(-1j * 4 * np.pi * ranges / self.wavelength)
            partial_image += val * phase_correction
            
            # Interpolate phase history values
            # for i in range(X_grid.shape[0]):
            #     for j in range(X_grid.shape[1]):
            #         if valid_mask[i, j]:
            #             idx = range_indices[i, j]
            #             idx_low = int(np.floor(idx))
            #             idx_high = int(np.ceil(idx))
            #             alpha = idx - idx_low
                        
            #             # Linear interpolation
            #             if idx_high < self.num_range_bins:
            #                 val = (1 - alpha) * self.cphd[pulse_idx, idx_low] + \
            #                       alpha * self.cphd[pulse_idx, idx_high]
            #             else:
            #                 val = self.cphd[pulse_idx, idx_low]
                        
            #             # Apply phase correction for range
            #             phase_correction = np.exp(-1j * 4 * np.pi * ranges[i, j] / self.wavelength)
            #             partial_image[i, j] += val * phase_correction
            
        # print("\nRange check:")
        # print(f"  Range bins: {self.range_bins[0]:.1f} to {self.range_bins[-1]:.1f} m")
        # print(f"  Computed ranges to grid: {np.min(ranges):.1f} to {np.max(ranges):.1f} m")
        return partial_image


# Example usage
if __name__ == "__main__":
    # Load your data
    cphd_file = 'ICEYE_X34_CPHD_SLH_951662092_20251001T181929.cphd'  # Shape: (28318, 34168)

    cphd_data,params,reader = sar_reader(cphd_file)

    pulse_analysis(cphd_data,reader,'single')
    
    # Initialize backprojection
    # bp = SARBackprojection(cphd_data, params, reader)
    
    # # Display
    # plt.figure(figsize=(12, 8))
    # plt.plot(20*np.log10(np.abs(fft_result)+1e-10)), 
    #         #aspect='auto', cmap='gray', vmin=-40, vmax=0)
    # plt.xlabel('Range Sample')
    # plt.ylabel('Doppler Frequency')
    # plt.title('Range-Doppler Map (First 1000 Pulses)')
    # #plt.colorbar(label='Magnitude (dB)')
    # plt.savefig('ffttest',dpi=150,bbox_inches='tight')

    # # Create image grid (start small for testing)
    # print("Creating image grid...")
    # image_grid = bp.create_image_grid(image_size_m=500, pixel_spacing_m=1.0)
    
    # # Perform backprojection (use subset of pulses for testing)
    # print("Starting backprojection...")
    # pulse_subset = range(0, 28318, 2)  # Use every 10th pulse for faster testing
    # image = bp.backproject(image_grid, None, n_workers=3, use_process=False)

    # # Check if trajectory is reasonable
    # velocity = np.diff(bp.sensor_positions, axis=0) / np.diff(bp.pulse_times)[:, np.newaxis]
    # speed = np.linalg.norm(velocity, axis=1)
    # print(f"Platform speed: {np.mean(speed):.1f} m/s (should be ~7500 m/s for satellite)")
    
    # # Display result
    # plt.figure(figsize=(10, 8))
    # plt.imshow(np.abs(image), cmap='gray', aspect='auto')
    # plt.colorbar(label='Magnitude')
    # plt.title('SAR Image - Backprojection Result')
    # plt.xlabel('Range')
    # plt.ylabel('Azimuth')
    # plt.tight_layout()
    # plt.savefig('sar_backprojection_result.png', dpi=150)
    # plt.show()
    
    # print("Backprojection complete!")