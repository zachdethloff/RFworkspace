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

def pulse_analysis(cphd_data,reader,test,pulse_idx):

    toa2 = reader.read_pvp_variable('TOA2', index=0, the_range=None)
    if test == 'single':
        print('Starting single pulse')
        test_data = cphd_data[pulse_idx, :]
        toa2 = toa2[pulse_idx]
        t_spacing = np.linspace(0, toa2, len(test_data))
        t_us = t_spacing * 1e6
        maskt = t_us < 50

        freqs = np.fft.fftshift(np.fft.fftfreq(len(test_data), 1/params['sample_rate']))
        freqs_mhz = freqs / 1e6
        powerf = 20*np.log10(np.abs(test_data)+1e-10)

        print('Starting Plots')

        plt.figure(figsize=(12, 5))
        plt.plot(freqs_mhz[::100], powerf[::100])
                #aspect='auto', cmap='gray', vmin=-40, vmax=0)
        plt.xlabel('Frequency Mhz')
        plt.ylabel('Power (dB)')
        plt.title('Single Pulse Frequency Domain')
        #plt.colorbar(label='Magnitude (dB)')
        plt.savefig('f_domain_pulse.png',dpi=300,bbox_inches='tight')

        # FFT in azimuth direction (along pulses)
        fft_result = np.fft.fftshift(np.fft.fft(test_data, axis=0), axes=0)
        powert = 20*np.log10(np.abs(fft_result)+1e-10)
        powert[~maskt] = 0

        plt.figure(figsize=(12, 8))
        plt.plot(t_us[::100], powert[::100])
                #aspect='auto', cmap='gray', vmin=-40, vmax=0)
        plt.xlabel('Time us')
        plt.ylabel('Power (dB)')
        plt.title('Single Pulse Time Domain')
        #plt.colorbar(label='Magnitude (dB)')
        plt.savefig('t_domain_pulse.png',dpi=300,bbox_inches='tight')


        print('Single Pulse complete')
    else:
        print('Starting batch test, all range bins')
        test_data = cphd_data[:pulse_idx, :]
        test_power = 20*np.log10(np.abs(test_data)+1e-10)
        low = np.percentile(test_power,20)
        high = np.percentile(test_power,99.9)
        range_check = [1000,2000,3000]

        fft_2d = np.fft.fftshift(np.fft.fft2(test_data))
        fft_check = np.fft.ifftshift(np.fft.ifft(test_data,axis=1),axes=1)
        fft_check = np.fft.fftshift(np.fft.fft(fft_check,axis=0),axes=0)
        rescale_2d = np.abs(fft_2d)**.5
        # rescale_check = np.abs(fft_check)**.5

        print('Starting Plots')
        plt.figure()
        plt.imshow(20*np.log10(np.abs(test_data)+1e-10),aspect='auto',cmap='gray',vmin=low,vmax=high)
        plt.xlabel('Range')
        plt.title(f'Pure Test Data Up To Pulse No. {pulse_idx}')
        plt.ylabel('Pulse Number')
        plt.savefig('Uncompressed.png',dpi=300,bbox_inches='tight')


        plt.figure(figsize=(12,8))
        #plt.plot(np.abs(fft_2d[:,range_check]))
        plt.plot(np.abs(fft_2d[range_check[0],:])**.5,label=range_check[0])
        plt.plot(np.abs(fft_2d[range_check[1],:])**.5,label=range_check[1])
        plt.plot(np.abs(fft_2d[range_check[2],:])**.5,label=range_check[2])
        plt.xlabel('Range Number')
        plt.title(f'All Ranges for Pulses {range_check}')
        plt.ylabel('Magnitude (dB)')
        plt.legend()
        plt.savefig('Single_Pulse.png',dpi=300,bbox_inches='tight')


        vmin = np.percentile(rescale_2d,5)
        vmax = np.percentile(rescale_2d,99.9)

        plt.figure(figsize=(12,8))
        plt.imshow(rescale_2d,aspect='auto',cmap='gray',vmin=vmin,vmax=vmax)
        plt.xlabel('Range Sample')
        plt.ylabel('Pulse Number')
        plt.title(f'Range-Doppler Map (First {pulse_idx} Pulses)')
        plt.colorbar(label='Magnitude (dB)')
        plt.savefig('2d_comp.png',dpi=300,bbox_inches='tight')

        # plt.figure(figsize=(12,8))
        # plt.imshow(rescale_check,aspect='auto',cmap='gray',vmin=vmin,vmax=vmax)
        # plt.xlabel('Range Sample')
        # plt.ylabel('Pulse Number')
        # plt.title(f'Range-Doppler Map (First {pulse_idx} Pulses)')
        # plt.colorbar(label='Magnitude (dB)')
        # plt.savefig('2d_check.png',dpi=300,bbox_inches='tight')


        doppler_slice = slice(1000, 4000)  # Around the peaks
        range_slice = slice(10000, 20000)  # Around bin 17000

        plt.figure(figsize=(12, 8))
        plt.imshow(rescale_2d[:, range_slice], 
                aspect='auto', cmap='grey', vmin=vmin, vmax=vmax)
        plt.xlabel('Range Sample')
        plt.ylabel('Pulse Number')
        plt.title('Zoomed Range-Doppler Map (Strong Signal Region)')
        plt.colorbar(label='Power (dB)')
        plt.savefig('Azimuth_zoom.png',dpi=300,bbox_inches='tight')



        print('Finished Plotting')

    

class SARBackprojection:
    def __init__(self, cphd_data, params, reader):
        """
        Initialize SAR Backprojection processor
        
        Args:
            cphd_data: Complex phase history data (num_pulses, num_range_bins)
            range_bins: Range values for each bin (meters)
            sicd_params: Dictionary with SICD parameters
        """
        c = 3e8
        self.cphd = np.fft.ifft(cphd_data,axis=1)
        self.num_pulses, self.num_range_bins = cphd_data.shape
        
        self.params = params
        self.params = params['sample_rate']
        self.center_freq = params['center_freq'] # Frequency Center
        self.wavelength = c / self.center_freq
        self.scp = params['scp']  # Scene center point
        self.collect_duration = params['collection_duration']
        self.scp_time = params['scp_time'] # Time at SCP
        self.row_uvect = params['row_uvect'] # Pixel range spacing
        self.col_uvect = params['col_uvect'] # Pixel azimuth spacing

        self.sensor_positions = reader.read_pvp_variable('TxPos', index=0,the_range=None)
        self.pulse_times = reader.read_pvp_variable('TxTime',index=0,the_range=None)

        toa_min = 0.0
        toa_max = reader.read_pvp_variable('TOA2',index=0,the_range=None)[0]
        srp_pos = reader.read_pvp_variable('SRPPos',index=0,the_range=None)
        ref_pulse_idx = len(self.sensor_positions) // 2
        ref_range = np.linalg.norm(self.sensor_positions[ref_pulse_idx] - srp_pos[ref_pulse_idx])
        toa_samples = np.linspace(toa_min,toa_max,self.num_range_bins)
        self.range_bins = ref_range + (c * toa_samples/2)
        print('Time of Arrival: ', toa_max)

        print(f"Range bins: {self.range_bins[0]:.1f} to {self.range_bins[-1]:.1f} m")
        print(f"Sensor to SCP range: ~560,000 m")
        print(f"Do they overlap? {self.range_bins[0] < 560000 < self.range_bins[-1]}")


        print("CPHD data consolidated")
        
    
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

    pulse_analysis(cphd_data,reader,'all',4000)
    
    # Initialize backprojection
    bp = SARBackprojection(cphd_data, params, reader)


    # # Create image grid (start small for testing)
    print("Creating image grid...")
    #image_grid = bp.create_image_grid(image_size_m=1000, pixel_spacing_m=1.0)
    test_grid = (np.array([[params['scp'][0]]]), 
            np.array([[params['scp'][1]]]), 
            np.array([[params['scp'][2]]]))

    image = bp.backproject(test_grid, pulse_subset=range(0, 4000))
    print(f"SCP pixel magnitude: {np.abs(image[0,0]):.1f}")
    print("Image grid successfully constructed")
    
    # # Perform backprojection (use subset of pulses for testing)
    # print("Starting backprojection...")
    # pulse_subset = range(0, 4000)  # Use every 10th pulse for faster testing
    # image = bp.backproject(image_grid, pulse_subset, n_workers=3, use_process=False)

    # # # Display result
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