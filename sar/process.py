from sarpy.io.phase_history.converter import open_phase_history
import numpy as np
import matplotlib.pyplot as plt
import sarpy.processing
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count


def sar_reader(
        cphd,sicd,rtp
):

    range_to_pixel = np.load(rtp)

    tree = ET.parse(sicd)
    root = tree.getroot()

    # Namespace (important for SICD)
    ns = {'sicd': 'urn:SICD:1.3.0'}

    position_elem = root.find('.//sicd:Position', ns)

    arp_poly_x = []
    arp_poly_y = []
    arp_poly_z = []

    if position_elem is not None:
        print("Position element found!")
        
        # Print all sub-elements to see structure
        for child in position_elem:
            print(f"Child tag: {child.tag}")
            print(f"Child text: {child.text}")
            
            # If it's ARPPoly, dive deeper
            if 'ARPPoly' in child.tag:
                print("\n=== ARPPoly Structure ===")
                for axis in child:  # X, Y, Z
                    coord = axis.tag[-1]
                    print(f"\nAxis: {coord}")
                    for coef in axis:
                        #exponent = coef.get('exponent1', '0')
                        value = coef.text
                        if coord == 'X':
                            arp_poly_x.append(value)
                        elif coord =='Y':
                            arp_poly_y.append(value)
                        else:
                            arp_poly_z.append(value)
    else:
        print("Position element not found!")

    scp_elem = float(root.find('.//sicd:SCPCOA/sicd:SCPTime', ns).text) 

    print(f"X:{len(arp_poly_x)} Y:{len(arp_poly_y)} Z:{len(arp_poly_z)}")
    print("If previous numbers don't match, quit now!")
    time.sleep(5)

    # 2. Get center frequency
    tx_freq = root.find('.//sicd:RadarCollection/sicd:TxFrequency/sicd:Min', ns)
    center_freq = float(tx_freq.text)  # in Hz

    # 3. Get timing information
    collect_start = root.find('.//sicd:Timeline/sicd:CollectStart', ns).text.replace('T',' ')
    collect_duration = float(root.find('.//sicd:Timeline/sicd:CollectDuration', ns).text)
    print(f'Image collection started at:  {collect_start.replace('Z','')} UTC with a duration of: {collect_duration:.2f} seconds\n')

    # 4. Get SCP (Scene Center Point)
    scp_ecf = root.find('.//sicd:GeoData/sicd:SCP/sicd:ECF', ns)
    scp_x = float(scp_ecf.find('sicd:X', ns).text)
    scp_y = float(scp_ecf.find('sicd:Y', ns).text)
    scp_z = float(scp_ecf.find('sicd:Z', ns).text)

        # Look for these in the SICD XML
    #grid = root.find('.//sicd:Grid', ns)

    # Row direction (typically range)
    row_uvect = [float(root.find('.//sicd:Grid/sicd:Row/sicd:UVectECF/sicd:X', ns).text), 
                 float(root.find('.//sicd:Grid/sicd:Row/sicd:UVectECF/sicd:Y', ns).text), 
                 float(root.find('.//sicd:Grid/sicd:Row/sicd:UVectECF/sicd:Z', ns).text)]

    # Col direction (typically azimuth)
    col_uvect = [float(root.find('.//sicd:Grid/sicd:Col/sicd:UVectECF/sicd:X', ns).text), 
                 float(root.find('.//sicd:Grid/sicd:Col/sicd:UVectECF/sicd:Y', ns).text), 
                 float(root.find('.//sicd:Grid/sicd:Col/sicd:UVectECF/sicd:Z', ns).text)]

    print('Major Image Parameters\n')

    print(f"Center Frequency: {center_freq/1e9} GHz")
    print(f"SCP: [{scp_x}, {scp_y}, {scp_z}]")

    params = {
        'arp_poly_x' : np.array(arp_poly_x),
        'arp_poly_y' : np.array(arp_poly_y),
        'arp_poly_z' : np.array(arp_poly_z),
        'center_freq' : center_freq,
        'scp' : np.array([scp_x,scp_y,scp_z]),
        'collection_start' : collect_start,
        'collection_duration' : collect_duration,
        'scp_time' : scp_elem,
        'row_uvect' : row_uvect,
        'col_uvect' : col_uvect
    }

    reader = open_phase_history(cphd)

    print('reader type = {}'.format(type(reader)))  # see the explicit reader type

    print('image size = {}\n'.format(reader.data_size))


    signal_data = reader.read_signal_block()
    signal_data = signal_data['spot_0_burst_0']

    return signal_data, params, range_to_pixel
    

class SARBackprojection:
    def __init__(self, cphd_data, sicd_params, range_bins):
        """
        Initialize SAR Backprojection processor
        
        Args:
            cphd_data: Complex phase history data (num_pulses, num_range_bins)
            range_bins: Range values for each bin (meters)
            sicd_params: Dictionary with SICD parameters
        """
        self.cphd = cphd_data
        self.range_bins = range_bins
        self.num_pulses, self.num_range_bins = cphd_data.shape
        
        # Extract SICD parameters
        self.arp_poly_x = sicd_params['arp_poly_x'].astype(float)
        self.arp_poly_y = sicd_params['arp_poly_y'].astype(float)
        self.arp_poly_z = sicd_params['arp_poly_z'].astype(float)
        self.center_freq = sicd_params['center_freq']
        self.wavelength = 3e8 / self.center_freq
        self.scp = sicd_params['scp']  # Scene center point
        self.collect_start = sicd_params['collection_start']
        self.collect_duration = sicd_params['collection_duration']
        self.scp_time = sicd_params['scp_time']
        self.row_uvect = sicd_params['row_uvect']
        self.col_uvect = sicd_params['col_uvect']
        
        # Compute time for each pulse
        self.pulse_times = np.linspace(0, self.collect_duration, self.num_pulses)
        
        # Precompute sensor positions for all pulses
        self.sensor_positions = self._compute_sensor_positions()

        # At SCP, what range does the sensor see?
        scp_pulse_idx = int(len(self.pulse_times) * self.scp_time / self.collect_duration)
        sensor_pos_at_scp = self.sensor_positions[scp_pulse_idx]
        range_to_scp = np.linalg.norm(sensor_pos_at_scp - self.scp)

        print(f"Computed range to SCP: {range_to_scp:.1f} m")
        print(f"Range bins span: {range_bins[0]:.1f} to {range_bins[-1]:.1f} m")
        print(f"Middle range bin: {range_bins[len(range_bins)//2]:.1f} m")
        print(f"Expected from SICD SlantRange: ???")  # Check your SICD XML
        
    def _compute_sensor_positions(self):
        """Compute sensor position for each pulse using ARPPoly"""
        positions = np.zeros((self.num_pulses, 3))
        
        for i, t in enumerate(self.pulse_times):
            # Evaluate polynomial: pos = c0 + c1*t + c2*t^2 + ...

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
        
        # Create local coordinate offsets
        row_offsets = np.linspace(-image_size_m/2, image_size_m/2, n_pixels)
        col_offsets = np.linspace(-image_size_m/2, image_size_m/2, n_pixels)
        Col_grid, Row_grid = np.meshgrid(col_offsets, row_offsets)
        
        # Get unit vectors from SICD (if available)
        row_uvect = self.row_uvect
        col_uvect = self.col_uvect
        
        if row_uvect is None or col_uvect is None:
            # Fallback: compute from geometry (old method)
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
        
        # Build 3D grid using SICD vectors
        # Row direction (typically range)
        # Col direction (typically azimuth)
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
        # single_pulse_image = bp.process_pulse_chunk([0], X_grid, Y_grid, Z_grid)
        # plt.figure()
        # plt.imshow(np.abs(single_pulse_image), cmap='gray')
        # plt.title('Single Pulse Contribution')
        # plt.show()
            
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
                print(f"  Processing pulse {pulse_idx}/{len(pulse_indices)}")
            
            # Get sensor position for this pulse
            sensor_pos = self.sensor_positions[pulse_idx]
            
            # Compute range from sensor to each pixel
            dx = X_grid - sensor_pos[0]
            dy = Y_grid - sensor_pos[1]
            dz = Z_grid - sensor_pos[2]
            ranges = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Find corresponding range bin for each pixel
            # Use linear interpolation
            range_indices = np.interp(ranges, self.range_bins, 
                                     np.arange(len(self.range_bins)))
            
            # Get complex values from phase history
            # Clip indices to valid range
            valid_mask = (range_indices >= 0) & (range_indices < self.num_range_bins - 1)
            
            # Interpolate phase history values
            for i in range(X_grid.shape[0]):
                for j in range(X_grid.shape[1]):
                    if valid_mask[i, j]:
                        idx = range_indices[i, j]
                        idx_low = int(np.floor(idx))
                        idx_high = int(np.ceil(idx))
                        alpha = idx - idx_low
                        
                        # Linear interpolation
                        if idx_high < self.num_range_bins:
                            val = (1 - alpha) * self.cphd[pulse_idx, idx_low] + \
                                  alpha * self.cphd[pulse_idx, idx_high]
                        else:
                            val = self.cphd[pulse_idx, idx_low]
                        
                        # Apply phase correction for range
                        phase_correction = np.exp(1j * 4 * np.pi * ranges[i, j] / self.wavelength)
                        partial_image[i, j] += val * phase_correction
        # print("\nRange check:")
        # print(f"  Range bins: {self.range_bins[0]:.1f} to {self.range_bins[-1]:.1f} m")
        # print(f"  Computed ranges to grid: {np.min(ranges):.1f} to {np.max(ranges):.1f} m")
        return partial_image


# Example usage
if __name__ == "__main__":
    # Load your data
    cphd_file = 'ICEYE_X34_CPHD_SLH_951662092_20251001T181929.cphd'  # Shape: (28318, 34168)
    range_file = 'range_to_pixel.npy' # Range values in meters
    sicd = 'ICEYE_X34_SICD_SLH_951662092_20251001T181932.xml'

    cphd_data,sicd_params,range_bins = sar_reader(cphd_file,sicd,range_file)
    
    # Initialize backprojection
    bp = SARBackprojection(cphd_data, sicd_params, range_bins)

    # Create image grid (start small for testing)
    print("Creating image grid...")
    image_grid = bp.create_image_grid(image_size_m=200, pixel_spacing_m=1.0)
    
    # Perform backprojection (use subset of pulses for testing)
    print("Starting backprojection...")
    pulse_subset = range(0, 28318, 2)  # Use every 10th pulse for faster testing
    image = bp.backproject(image_grid, None, n_workers=8, use_process=False)

    # fig = plt.figure(figsize=(12, 4))

    # plt.subplot(131)
    # plt.plot(bp.pulse_times, bp.sensor_positions[:, 0])
    # plt.xlabel('Time (s)')
    # plt.ylabel('X (m)')
    # plt.title('X Position vs Time')

    # plt.subplot(132)
    # plt.plot(bp.pulse_times, bp.sensor_positions[:, 1])
    # plt.xlabel('Time (s)')
    # plt.ylabel('Y (m)')
    # plt.title('Y Position vs Time')

    # plt.subplot(133)
    # plt.plot(bp.pulse_times, bp.sensor_positions[:, 2])
    # plt.xlabel('Time (s)')
    # plt.ylabel('Z (m)')
    # plt.title('Z Position vs Time')

    # plt.tight_layout()
    # plt.show()

    # Check if trajectory is reasonable
    velocity = np.diff(bp.sensor_positions, axis=0) / np.diff(bp.pulse_times)[:, np.newaxis]
    speed = np.linalg.norm(velocity, axis=1)
    print(f"Platform speed: {np.mean(speed):.1f} m/s (should be ~7500 m/s for satellite)")
    
    # Display result
    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(image), cmap='gray', aspect='auto')
    plt.colorbar(label='Magnitude')
    plt.title('SAR Image - Backprojection Result')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.tight_layout()
    plt.savefig('sar_backprojection_result.png', dpi=150)
    plt.show()
    
    print("Backprojection complete!")