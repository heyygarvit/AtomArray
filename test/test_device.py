import unittest
from unittest.mock import patch, Mock
import sys
sys.path.append('src')
import core.device  # Replace with the name of the module containing the provided code

class TestDevice(unittest.TestCase):

    def test_is_cuda(self):
        device = core.device.Device("cuda")
        self.assertTrue(device.is_cuda())

    def test_is_metal(self):
        device = core.device.Device("metal")
        self.assertTrue(device.is_metal())

    def test_is_cpu(self):
        device = core.device.Device("cpu")
        self.assertTrue(device.is_cpu())

class TestDefaultDevice(unittest.TestCase):

    @patch('core.device.has_nvidia_gpu', return_value=True)
    def test_get_default_device_cuda(self, mock_gpu):
        self.assertEqual(core.device.get_default_device().device_type, "cuda")

    @patch('core.device.has_nvidia_gpu', return_value=False)
    @patch('core.device.is_mac', return_value=True)
    @patch('core.device.supports_metal', return_value=True)
    def test_get_default_device_metal(self, mock_gpu, mock_mac, mock_metal):
        self.assertEqual(core.device.get_default_device().device_type, "metal")

    @patch('core.device.has_nvidia_gpu', return_value=False)
    @patch('core.device.is_mac', return_value=False)
    def test_get_default_device_cpu(self, mock_gpu, mock_mac):
        self.assertEqual(core.device.get_default_device().device_type, "cpu")

    @patch('core.device.subprocess.check_output', return_value=b'NVIDIA')
    def test_has_nvidia_gpu_true(self, mock_output):
        self.assertTrue(core.device.has_nvidia_gpu())

    @patch('core.device.subprocess.check_output', side_effect=FileNotFoundError)
    def test_has_nvidia_gpu_false(self, mock_output):
        self.assertFalse(core.device.has_nvidia_gpu())

    @patch('core.device.platform.system', return_value="Darwin")
    def test_is_mac_true(self, mock_system):
        self.assertTrue(core.device.is_mac())

    @patch('core.device.platform.system', return_value="Windows")
    def test_is_mac_false(self, mock_system):
        self.assertFalse(core.device.is_mac())

    @patch('core.device.platform.mac_ver', return_value=("10.15.0", ("", "", ""), ""))
    def test_supports_metal_true(self, mock_mac_ver):
        self.assertTrue(core.device.supports_metal())

    @patch('core.device.platform.mac_ver', return_value=("10.13.0", ("", "", ""), ""))
    def test_supports_metal_false(self, mock_mac_ver):
        self.assertFalse(core.device.supports_metal())

if __name__ == "__main__":
    unittest.main()
