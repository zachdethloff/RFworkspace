from sarpy.io.phase_history.converter import open_phase_history
import numpy as np
import matplotlib.pyplot as plt
import sarpy.processing
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
    

class SARBackprojection:
    def __init__(self, cphd_data, range_bins, sicd_params):
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
        self.arp_poly_x = sicd_params['arp_poly_x']
        self.arp_poly_y = sicd_params['arp_poly_y']
        self.arp_poly_z = sicd_params['arp_poly_z']
        self.center_freq = sicd_params['center_freq']
        self.wavelength = 3e8 / self.center_freq
        self.scp = sicd_params['scp']  # Scene center point
        self.collect_start = sicd_params['collect_start']
        self.collect_duration = sicd_params['collect_duration']
        
        # Compute time for each pulse
        self.pulse_times = np.linspace(0, self.collect_duration, self.num_pulses)
        
        # Precompute sensor positions for all pulses
        self.sensor_positions = self._compute_sensor_positions()

    def sar_reader(self, size = 'full'):

        reader = open_phase_history('ICEYE_X34_CPHD_SLH_951662092_20251001T181929.cphd')

        print('reader type = {}'.format(type(reader)))  # see the explicit reader type

        print('image size = {}'.format(reader.data_size))


        signal_data = reader.read_signal_block()
        signal_data = signal_data['spot_0_burst_0']

        range_to_pixel = np.load('range_to_pixel.npy')
        print("Shape:", range_to_pixel.shape)

        tree = ET.parse('ICEYE_X34_SICD_SLH_951662092_20251001T181932.xml')
        root = tree.getroot()

        # Namespace (important for SICD)
        ns = {'sicd': 'urn:SICD:1.3.0'}

        position_elem = root.find('.//sicd:Position', ns)

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
                        print(f"\nAxis: {axis.tag}")
                        for coef in axis:
                            exponent = coef.get('exponent1', '0')
                            value = coef.text
                            print(f"  Coef (t^{exponent}): {value}")
        else:
            print("Position element not found!")

        # Alternative: Just print the entire Position section as text
        print("\n=== Raw XML for Position ===")
        position_str = ET.tostring(position_elem, encoding='unicode')
        print(position_str)

        # 2. Get center frequency
        tx_freq = root.find('.//sicd:RadarCollection/sicd:TxFrequency/sicd:Min', ns)
        center_freq = float(tx_freq.text)  # in Hz

        # 3. Get timing information
        collect_start = root.find('.//sicd:Timeline/sicd:CollectStart', ns).text
        collect_duration = float(root.find('.//sicd:Timeline/sicd:CollectDuration', ns).text)
        print(collect_start,collect_duration)

        # 4. Get SCP (Scene Center Point)
        scp_ecf = root.find('.//sicd:GeoData/sicd:SCP/sicd:ECF', ns)
        scp_x = float(scp_ecf.find('sicd:X', ns).text)
        scp_y = float(scp_ecf.find('sicd:Y', ns).text)
        scp_z = float(scp_ecf.find('sicd:Z', ns).text)

        print(f"Center Frequency: {center_freq/1e9} GHz")
        print(f"SCP: [{scp_x}, {scp_y}, {scp_z}]")

        return signal_data
        
    def _compute_sensor_positions(self):
        """Compute sensor position for each pulse using ARPPoly"""
        positions = np.zeros((self.num_pulses, 3))
        
        for i, t in enumerate(self.pulse_times):
            # Evaluate polynomial: pos = c0 + c1*t + c2*t^2 + ...
            positions[i, 0] = np.polyval(self.arp_poly_x[::-1], t)  # X
            positions[i, 1] = np.polyval(self.arp_poly_y[::-1], t)  # Y
            positions[i, 2] = np.polyval(self.arp_poly_z[::-1], t)  # Z
            
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
        x_local = np.linspace(-image_size_m/2, image_size_m/2, n_pixels)
        y_local = np.linspace(-image_size_m/2, image_size_m/2, n_pixels)
        X_local, Y_local = np.meshgrid(x_local, y_local)
        
        # Get average sensor position to define image plane orientation
        avg_sensor_pos = np.mean(self.sensor_positions, axis=0)
        
        # Vector from SCP to sensor
        look_vec = avg_sensor_pos - self.scp
        look_vec = look_vec / np.linalg.norm(look_vec)
        
        # Create orthogonal basis for image plane
        # Ground plane normal (approximate)
        ground_normal = self.scp / np.linalg.norm(self.scp)
        
        # Range direction (perpendicular to velocity, in ground plane)
        range_dir = np.cross(ground_normal, look_vec)
        range_dir = range_dir / np.linalg.norm(range_dir)
        
        # Azimuth direction
        az_dir = np.cross(range_dir, ground_normal)
        az_dir = az_dir / np.linalg.norm(az_dir)
        
        # Build 3D grid
        X_ecef = self.scp[0] + X_local[..., np.newaxis] * az_dir[0] + Y_local[..., np.newaxis] * range_dir[0]
        Y_ecef = self.scp[1] + X_local[..., np.newaxis] * az_dir[1] + Y_local[..., np.newaxis] * range_dir[1]
        Z_ecef = self.scp[2] + X_local[..., np.newaxis] * az_dir[2] + Y_local[..., np.newaxis] * range_dir[2]
        
        return X_ecef.squeeze(), Y_ecef.squeeze(), Z_ecef.squeeze()
    
    def backproject(self, image_grid, pulse_subset=None):
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
        
        print(f"Backprojecting {len(pulse_indices)} pulses...")
        
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
                        phase_correction = np.exp(-1j * 4 * np.pi * ranges[i, j] / self.wavelength)
                        image[i, j] += val * phase_correction
        
        return image


# Example usage
if __name__ == "__main__":
    # Load your data
    cphd_data = np.load('cphd_data.npy')  # Shape: (28318, 34168)
    range_bins = np.load('range_to_pixel.npy')  # Range values in meters
    
    # SICD parameters from your XML
    sicd_params = {
        'arp_poly_x': np.array([2478849.60868615, -572.7557825700707, -1.892184847718151,
                                -2.518375956724047e-05, 2.475272494045103e-07,
                                -4.418402022842363e-09, 1.0048657958395474e-09,
                                -1.2315954197005195e-10, 6.2764554779468074e-12]),
        'arp_poly_y': np.array([4902985.249702282, -4735.895094222927, -3.016243420162082,
                                0.0010818181782065305, 3.207745334016667e-07,
                                -6.09022185265171e-09, 1.388446627818461e-09,
                                -1.759891919990173e-10, 9.167253767397782e-12]),
        'arp_poly_z': np.array([4048509.5377419027, 6073.364848091458, -2.547760010476697,
                                -0.001268370320138367, 3.1657466936016516e-07,
                                -1.2541545425691035e-08, 3.0327964266719605e-09,
                                -3.821507398468962e-10, 1.9914932572697033e-11]),
        'center_freq': 9.5e9,  # 9.5 GHz
        'scp': np.array([2604951.366402925, 4444849.345567337, 3749150.110799195]),
        'collect_start': '2025-10-01T18:19:32.122111Z',
        'collect_duration': 4.521394176008343
    }
    
    # Initialize backprojection
    bp = SARBackprojection(cphd_data, range_bins, sicd_params)
    
    # Create image grid (start small for testing)
    print("Creating image grid...")
    image_grid = bp.create_image_grid(image_size_m=500, pixel_spacing_m=2.0)
    
    # Perform backprojection (use subset of pulses for testing)
    print("Starting backprojection...")
    pulse_subset = range(0, 28318, 10)  # Use every 10th pulse for faster testing
    image = bp.backproject(image_grid, pulse_subset=pulse_subset)
    
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