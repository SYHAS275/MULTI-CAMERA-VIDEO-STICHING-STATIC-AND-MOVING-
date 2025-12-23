#!/usr/bin/env python3
"""
GPU-Accelerated Video Panorama Stitcher
Single-file version combining video processing pipeline with Stitcher class.

Features:
- NVDEC GPU video decoding
- CUDA-accelerated warping, color conversion, and blending
- LAB color space matching with luminance protection
- Parallax-aware blending
- NVENC GPU video encoding
"""

import numpy as np
import cv2
import subprocess
from datetime import datetime
import time
import os
from threading import Thread
from queue import Queue
import logging
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Get process handle for memory monitoring
PROCESS = psutil.Process(os.getpid())

# Initialize NVIDIA GPU monitoring
GPU_AVAILABLE = False
GPU_HANDLE = None
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    GPU_AVAILABLE = True
    logger.info("GPU monitoring enabled (pynvml)")
except ImportError:
    logger.debug("pynvml not installed - GPU monitoring disabled")
except Exception as e:
    logger.debug(f"GPU monitoring unavailable: {e}")


# =============================================================================
# STITCHER CLASS
# =============================================================================

class Stitcher:
    # Thresholds and parameters
    OVERLAP_THRESHOLD_RATIO = 0.15      # Initial threshold for detecting overlap region
    FALLBACK_THRESHOLD_RATIO = 0.05     # Fallback threshold if initial fails
    BLEND_WIDTH = 100                    # Width of the blending region in pixels
    BLACK_PIXEL_THRESHOLD = 10           # Pixels below this are considered black
    PARALLAX_DIFF_THRESHOLD = 15         # Difference threshold for parallax detection
    VALID_PIXEL_MIN = 50                 # Minimum value for valid pixel in color matching
    VALID_PIXEL_MAX = 220                # Maximum value for valid pixel in color matching
    MIN_VALID_PIXELS = 500               # Minimum valid pixels needed for color matching
    PERSPECTIVE_THRESHOLD = 0.005        # Threshold for perspective distortion (relaxed for moving camera)
    SCALE_MAX = 1.5                       # Maximum allowed scale factor (relaxed for moving camera)
    SCALE_MIN = 0.67                      # Minimum allowed scale factor (relaxed for moving camera)
    LUMINANCE_DARK_THRESHOLD = 60        # Dark luminance protection threshold
    LUMINANCE_BRIGHT_THRESHOLD = 200     # Bright luminance protection threshold
    DILATION_KERNEL_SIZE = 40            # Kernel size for parallax mask dilation
    
    # Moving camera parameters
    KEYFRAME_INTERVAL = 30               # Recalculate homography every N frames
    HOMOGRAPHY_CHANGE_THRESHOLD = 0.1    # Recalculate if homography changes more than this
    HOMOGRAPHY_BUFFER_SIZE = 5           # Number of homographies to average for smoothing
    MIN_MATCH_COUNT = 50                 # Minimum matches required, else force recalculation
    FEATURE_COUNT = 8000                 # Increased features for moving camera
    
    def __init__(self, use_cuda=True, moving_camera=False):
        self.cachedH = None
        self.blend_start = None
        self.blend_end = None
        self.output_width = None
        self.output_height = None
        self.crop_top = None
        self.crop_bottom = None
        
        # Moving camera support
        self.moving_camera = moving_camera
        self.frame_count = 0
        self.homography_buffer = []      # Buffer for temporal smoothing
        self.last_match_count = 0        # Track match quality
        
        # Check CUDA availability
        self.use_cuda = use_cuda and self._check_cuda()
        
        if self.use_cuda:
            self._init_cuda()
        else:
            logger.info("Running on CPU")
        
        # Feature detector - CPU SIFT (more reliable)
        # Use more features for moving camera
        n_features = self.FEATURE_COUNT if moving_camera else 5000
        self.detector = cv2.SIFT_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.use_gpu_features = False
        logger.info(f"Using CPU SIFT for feature detection (nfeatures={n_features})")
        
        if moving_camera:
            logger.info("Moving camera mode enabled - homography will be recalculated periodically")

    def _check_cuda(self):
        """Check if CUDA is available and working."""
        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
            if count > 0:
                test = cv2.cuda_GpuMat()
                test.upload(np.zeros((10, 10), dtype=np.uint8))
                return True
        except Exception as e:
            logger.debug(f"CUDA check failed: {e}")
        return False

    def _init_cuda(self):
        """Initialize CUDA resources."""
        logger.info(f"CUDA enabled! Devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
        cv2.cuda.printCudaDeviceInfo(0)
        
        # GPU matrices for reuse
        self.gpu_frame = cv2.cuda_GpuMat()
        self.gpu_frameB = cv2.cuda_GpuMat()
        self.gpu_warped = cv2.cuda_GpuMat()
        self.gpu_gray = cv2.cuda_GpuMat()
        self.gpu_grayB = cv2.cuda_GpuMat()
        self.gpu_result = cv2.cuda_GpuMat()
        self.gpu_temp = cv2.cuda_GpuMat()
        self.gpu_blend_left = cv2.cuda_GpuMat()
        self.gpu_blend_right = cv2.cuda_GpuMat()
        
        # CUDA stream for async operations
        self.stream = cv2.cuda_Stream()
        
        # Pre-create GPU filters
        try:
            self.gpu_gaussian = cv2.cuda.createGaussianFilter(
                cv2.CV_8UC1, cv2.CV_8UC1, (31, 31), 0
            )
            self.gpu_gaussian_small = cv2.cuda.createGaussianFilter(
                cv2.CV_8UC1, cv2.CV_8UC1, (11, 11), 0
            )
            self.has_gpu_filters = True
        except cv2.error as e:
            self.has_gpu_filters = False
            logger.debug(f"GPU Gaussian filter not available: {e}")
        
        logger.info("CUDA initialized - Full GPU acceleration enabled")

    def _should_recalculate_homography(self):
        """Determine if homography should be recalculated for moving camera."""
        if not self.moving_camera:
            return self.cachedH is None
        
        # Always calculate on first frame
        if self.cachedH is None:
            return True
        
        # Recalculate every N frames
        if self.frame_count % self.KEYFRAME_INTERVAL == 0:
            return True
        
        # Recalculate if match quality dropped
        if self.last_match_count < self.MIN_MATCH_COUNT:
            return True
        
        return False
    
    def _smooth_homography(self, H):
        """Apply temporal smoothing to homography for stable output."""
        self.homography_buffer.append(H.copy())
        
        # Keep buffer size limited
        if len(self.homography_buffer) > self.HOMOGRAPHY_BUFFER_SIZE:
            self.homography_buffer.pop(0)
        
        # If only one homography, return it
        if len(self.homography_buffer) == 1:
            return H
        
        # Weighted average (more recent = higher weight)
        weights = np.array([i + 1 for i in range(len(self.homography_buffer))], dtype=np.float32)
        weights /= weights.sum()
        
        smoothed_H = np.zeros_like(H, dtype=np.float64)
        for i, h in enumerate(self.homography_buffer):
            smoothed_H += weights[i] * h
        
        # Normalize
        smoothed_H /= smoothed_H[2, 2]
        
        return smoothed_H.astype(np.float32)
    
    def _homography_changed_significantly(self, new_H):
        """Check if new homography differs significantly from cached."""
        if self.cachedH is None:
            return True
        
        # Compare Frobenius norm of difference
        diff = np.linalg.norm(new_H - self.cachedH, 'fro')
        return diff > self.HOMOGRAPHY_CHANGE_THRESHOLD

    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
        (imageB, imageA) = images  # left = B, right = A
        
        self.frame_count += 1

        # Check if we need to recalculate homography
        if self._should_recalculate_homography():
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)

            M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB,
                                    ratio, reprojThresh)
            if M is None:
                logger.warning("Not enough matches to compute homography")
                # For moving camera, try to use last known good homography
                if self.moving_camera and self.cachedH is not None:
                    logger.info("Using last known homography")
                    return self.applyWarp(imageA, imageB, self.cachedH)
                return None

            (matches, H, status) = M
            self.last_match_count = len(matches)
            
            if H is None:
                logger.warning("Homography is None")
                if self.moving_camera and self.cachedH is not None:
                    return self.applyWarp(imageA, imageB, self.cachedH)
                return None

            H = self._constrainHomography(H, imageA.shape, imageB.shape)
            
            # For moving camera, apply smoothing
            if self.moving_camera:
                H = self._smooth_homography(H)
                
                # Reset blend region if homography changed significantly
                # BUT keep output_width and output_height fixed to avoid VideoWriter errors
                if self._homography_changed_significantly(H):
                    self.blend_start = None
                    self.blend_end = None
                    # Do NOT reset output_width/output_height - VideoWriter needs consistent size
                    logger.debug(f"Homography updated at frame {self.frame_count}")
            
            self.cachedH = H.astype(np.float32)

        return self.applyWarp(imageA, imageB, self.cachedH)
    
    def _constrainHomography(self, H, shapeA, shapeB):
        H = H / H[2, 2]

        if abs(H[2, 0]) > self.PERSPECTIVE_THRESHOLD or abs(H[2, 1]) > self.PERSPECTIVE_THRESHOLD:
            H[2, 0] *= 0.5
            H[2, 1] *= 0.5
            H = H / H[2, 2]

        scale_x = np.sqrt(H[0, 0]**2 + H[1, 0]**2)
        scale_y = np.sqrt(H[0, 1]**2 + H[1, 1]**2)

        if scale_x > self.SCALE_MAX or scale_x < self.SCALE_MIN:
            H[0, 0] /= scale_x
            H[1, 0] /= scale_x
        if scale_y > self.SCALE_MAX or scale_y < self.SCALE_MIN:
            H[0, 1] /= scale_y
            H[1, 1] /= scale_y

        H = H / H[2, 2]
        return H

    def applyWarp(self, imageA, imageB, H):
        """
        Fully CUDA-accelerated warp and blend.
        """
        h, w = imageB.shape[:2]
        
        # Calculate canvas size
        corners = np.float32([[0, 0], [imageA.shape[1], 0], 
                              [imageA.shape[1], imageA.shape[0]], [0, imageA.shape[0]]])
        warped_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H)
        max_x = int(np.max(warped_corners[:, 0, 0]))
        canvas_width = min(max_x + 50, imageA.shape[1] + imageB.shape[1])

        # === GPU WARP ===
        if self.use_cuda:
            try:
                self.gpu_frame.upload(imageA, self.stream)
                gpu_warped = cv2.cuda.warpPerspective(
                    self.gpu_frame, H,
                    (canvas_width, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                    stream=self.stream
                )
                
                # GPU color conversion for grayscale
                gpu_warped_gray = cv2.cuda.cvtColor(gpu_warped, cv2.COLOR_BGR2GRAY, stream=self.stream)
                
                self.stream.waitForCompletion()
                warped = gpu_warped.download()
                warped_gray = gpu_warped_gray.download()
            except Exception as e:
                logger.debug(f"CUDA warp failed: {e}, using CPU")
                warped = cv2.warpPerspective(imageA, H, (canvas_width, h),
                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            warped = cv2.warpPerspective(imageA, H, (canvas_width, h),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Find blend region (first frame only)
        if self.blend_start is None:
            overlap_start = w
            threshold_pixels = int(h * self.OVERLAP_THRESHOLD_RATIO)

            for x in range(w):
                if np.count_nonzero(warped_gray[:, x] > self.BLACK_PIXEL_THRESHOLD) >= threshold_pixels:
                    overlap_start = x
                    break

            if overlap_start >= w:
                threshold_pixels = int(h * self.FALLBACK_THRESHOLD_RATIO)
                for x in range(w):
                    if np.count_nonzero(warped_gray[:, x] > self.BLACK_PIXEL_THRESHOLD) >= threshold_pixels:
                        overlap_start = x
                        break

            if overlap_start > w - 30:
                overlap_start = max(0, w - self.BLEND_WIDTH)

            overlap_width = w - overlap_start
            
            blend_center = overlap_start + overlap_width // 2
            self.blend_start = max(0, blend_center - self.BLEND_WIDTH // 2)
            self.blend_end = min(w, blend_center + self.BLEND_WIDTH // 2)
            
            valid_cols = np.where(np.any(warped_gray > self.BLACK_PIXEL_THRESHOLD, axis=0))[0]
            if len(valid_cols) > 0:
                self.output_width = min(valid_cols[-1] + 10, canvas_width)
            else:
                self.output_width = canvas_width
            self.output_height = h
            
            # Pre-compute gradient mask on GPU
            actual_blend_width = self.blend_end - self.blend_start
            if actual_blend_width > 0:
                mask_1d = np.linspace(0, 1, actual_blend_width, dtype=np.float32)
                mask_1d = mask_1d * mask_1d * (3 - 2 * mask_1d)
                self.gradient_mask = np.tile(mask_1d, (h, 1))
                self.gradient_mask_3 = np.dstack([self.gradient_mask] * 3)
            
            logger.info(f"Blend region: {self.blend_start} to {self.blend_end}")
            logger.info(f"Output size: {self.output_width}x{self.output_height}")

        blend_start = self.blend_start
        blend_end = self.blend_end
        actual_blend_width = blend_end - blend_start

        result = warped.copy()

        # === GPU COLOR MATCHING ===
        sample_width = 150
        sample_start = max(0, blend_start - sample_width)
        sample_end = min(w, blend_end + sample_width)
        
        sample_left = imageB[:, sample_start:sample_end].copy()
        sample_right = warped[:h, sample_start:sample_end].copy()
        
        if self.use_cuda:
            try:
                self.gpu_temp.upload(sample_right, self.stream)
                gpu_sample_gray = cv2.cuda.cvtColor(self.gpu_temp, cv2.COLOR_BGR2GRAY, stream=self.stream)
                self.stream.waitForCompletion()
                right_gray_sample = gpu_sample_gray.download()
            except cv2.error:
                right_gray_sample = cv2.cvtColor(sample_right, cv2.COLOR_BGR2GRAY)
        else:
            right_gray_sample = cv2.cvtColor(sample_right, cv2.COLOR_BGR2GRAY)
        
        valid_mask = (right_gray_sample > self.VALID_PIXEL_MIN) & (right_gray_sample < self.VALID_PIXEL_MAX)
        
        if np.sum(valid_mask) > self.MIN_VALID_PIXELS:
            # GPU LAB conversion
            if self.use_cuda:
                try:
                    self.gpu_temp.upload(sample_left, self.stream)
                    gpu_left_lab = cv2.cuda.cvtColor(self.gpu_temp, cv2.COLOR_BGR2Lab, stream=self.stream)
                    self.gpu_temp.upload(sample_right, self.stream)
                    gpu_right_lab = cv2.cuda.cvtColor(self.gpu_temp, cv2.COLOR_BGR2Lab, stream=self.stream)
                    self.stream.waitForCompletion()
                    left_lab = gpu_left_lab.download().astype(np.float32)
                    right_lab = gpu_right_lab.download().astype(np.float32)
                except cv2.error:
                    left_lab = cv2.cvtColor(sample_left, cv2.COLOR_BGR2LAB).astype(np.float32)
                    right_lab = cv2.cvtColor(sample_right, cv2.COLOR_BGR2LAB).astype(np.float32)
            else:
                left_lab = cv2.cvtColor(sample_left, cv2.COLOR_BGR2LAB).astype(np.float32)
                right_lab = cv2.cvtColor(sample_right, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            transfer_params = []
            for c in range(3):
                left_vals = left_lab[:, :, c][valid_mask]
                right_vals = right_lab[:, :, c][valid_mask]
                
                if len(left_vals) > 100 and len(right_vals) > 100:
                    left_mean, left_std = np.mean(left_vals), np.std(left_vals)
                    right_mean, right_std = np.mean(right_vals), np.std(right_vals)
                    
                    if right_std > 1:
                        scale = np.clip(left_std / right_std, 0.8, 1.2)
                    else:
                        scale = 1.0
                    transfer_params.append((scale, right_mean, left_mean))
                else:
                    transfer_params.append((1.0, 0.0, 0.0))
            
            # Apply color correction
            if self.use_cuda:
                try:
                    self.gpu_temp.upload(warped, self.stream)
                    gpu_warped_lab = cv2.cuda.cvtColor(self.gpu_temp, cv2.COLOR_BGR2Lab, stream=self.stream)
                    self.stream.waitForCompletion()
                    warped_lab = gpu_warped_lab.download().astype(np.float32)
                except cv2.error:
                    warped_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
            else:
                warped_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            L_channel = warped_lab[:, :, 0]
            luminance_protection = np.ones_like(L_channel)
            
            dark_mask = L_channel < self.LUMINANCE_DARK_THRESHOLD
            luminance_protection[dark_mask] = L_channel[dark_mask] / float(self.LUMINANCE_DARK_THRESHOLD)
            
            bright_mask = L_channel > self.LUMINANCE_BRIGHT_THRESHOLD
            luminance_protection[bright_mask] = (255 - L_channel[bright_mask]) / (255.0 - self.LUMINANCE_BRIGHT_THRESHOLD)
            
            luminance_protection = np.clip(luminance_protection, 0.1, 1.0)
            
            for c in range(3):
                scale, right_mean, left_mean = transfer_params[c]
                corrected = (warped_lab[:, :, c] - right_mean) * scale + left_mean
                original = warped_lab[:, :, c]
                warped_lab[:, :, c] = original * (1 - luminance_protection) + corrected * luminance_protection
                warped_lab[:, :, c] = np.clip(warped_lab[:, :, c], 0, 255)
            
            # GPU LAB to BGR conversion
            if self.use_cuda:
                try:
                    self.gpu_temp.upload(warped_lab.astype(np.uint8), self.stream)
                    gpu_warped_bgr = cv2.cuda.cvtColor(self.gpu_temp, cv2.COLOR_Lab2BGR, stream=self.stream)
                    self.stream.waitForCompletion()
                    warped = gpu_warped_bgr.download()
                except cv2.error:
                    warped = cv2.cvtColor(warped_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            else:
                warped = cv2.cvtColor(warped_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # === GPU BLENDING ===
        result = warped.copy()
        result[:h, :blend_start] = imageB[:h, :blend_start]

        if actual_blend_width > 0:
            left_region = imageB[:, blend_start:blend_end].copy()
            right_region = result[:h, blend_start:blend_end].copy()
            
            # GPU grayscale for diff
            if self.use_cuda:
                try:
                    self.gpu_blend_left.upload(left_region, self.stream)
                    self.gpu_blend_right.upload(right_region, self.stream)
                    gpu_left_gray = cv2.cuda.cvtColor(self.gpu_blend_left, cv2.COLOR_BGR2GRAY, stream=self.stream)
                    gpu_right_gray = cv2.cuda.cvtColor(self.gpu_blend_right, cv2.COLOR_BGR2GRAY, stream=self.stream)
                    self.stream.waitForCompletion()
                    left_gray = gpu_left_gray.download().astype(np.float32)
                    right_gray = gpu_right_gray.download().astype(np.float32)
                except cv2.error:
                    left_gray = cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    right_gray = cv2.cvtColor(right_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                left_gray = cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
                right_gray = cv2.cvtColor(right_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            diff = np.abs(left_gray - right_gray)
            parallax_mask = (diff > self.PARALLAX_DIFF_THRESHOLD).astype(np.float32)
            parallax_mask = cv2.dilate(parallax_mask, np.ones((self.DILATION_KERNEL_SIZE, self.DILATION_KERNEL_SIZE), np.uint8))
            
            # GPU Gaussian blur for parallax mask
            if self.use_cuda and self.has_gpu_filters:
                try:
                    parallax_mask_u8 = (parallax_mask * 255).astype(np.uint8)
                    self.gpu_temp.upload(parallax_mask_u8, self.stream)
                    gpu_blurred = self.gpu_gaussian.apply(self.gpu_temp, stream=self.stream)
                    self.stream.waitForCompletion()
                    parallax_mask = gpu_blurred.download().astype(np.float32) / 255.0
                except cv2.error:
                    parallax_mask = cv2.GaussianBlur(parallax_mask, (31, 31), 0)
            else:
                parallax_mask = cv2.GaussianBlur(parallax_mask, (31, 31), 0)
            
            final_mask = self.gradient_mask.copy()
            final_mask[parallax_mask > 0.5] = 0.0
            
            # GPU Gaussian blur for final mask
            if self.use_cuda and self.has_gpu_filters:
                try:
                    final_mask_u8 = (final_mask * 255).astype(np.uint8)
                    self.gpu_temp.upload(final_mask_u8, self.stream)
                    gpu_blurred = self.gpu_gaussian_small.apply(self.gpu_temp, stream=self.stream)
                    self.stream.waitForCompletion()
                    final_mask = gpu_blurred.download().astype(np.float32) / 255.0
                except cv2.error:
                    final_mask = cv2.GaussianBlur(final_mask, (11, 11), 0)
            else:
                final_mask = cv2.GaussianBlur(final_mask, (11, 11), 0)
            
            final_mask_3 = np.dstack([final_mask] * 3)
            
            left_float = left_region.astype(np.float32)
            right_float = right_region.astype(np.float32)
            blended = left_float * (1.0 - final_mask_3) + right_float * final_mask_3
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            
            result[:h, blend_start:blend_end] = blended

        # Fill holes
        left_part = imageB[:h, :w]
        result_part = result[:h, :w]

        if self.use_cuda:
            try:
                self.gpu_temp.upload(left_part, self.stream)
                gpu_left_gray = cv2.cuda.cvtColor(self.gpu_temp, cv2.COLOR_BGR2GRAY, stream=self.stream)
                self.gpu_temp.upload(result_part, self.stream)
                gpu_result_gray = cv2.cuda.cvtColor(self.gpu_temp, cv2.COLOR_BGR2GRAY, stream=self.stream)
                self.stream.waitForCompletion()
                left_gray_full = gpu_left_gray.download()
                result_gray_full = gpu_result_gray.download()
            except cv2.error:
                left_gray_full = cv2.cvtColor(left_part, cv2.COLOR_BGR2GRAY)
                result_gray_full = cv2.cvtColor(result_part, cv2.COLOR_BGR2GRAY)
        else:
            left_gray_full = cv2.cvtColor(left_part, cv2.COLOR_BGR2GRAY)
            result_gray_full = cv2.cvtColor(result_part, cv2.COLOR_BGR2GRAY)

        holes = (result_gray_full < self.BLACK_PIXEL_THRESHOLD) & (left_gray_full > self.BLACK_PIXEL_THRESHOLD)
        if np.any(holes):
            holes_3 = np.dstack([holes] * 3)
            result_part[holes_3] = left_part[holes_3]

        result[:h, :w] = result_part
        
        result = self.fillFromSourceImages(result, imageA, imageB, H)
        
        # Crop to output size
        cropped = result[:self.output_height, :self.output_width]
        
        # For moving camera: ensure consistent output size by padding/cropping
        if self.moving_camera and hasattr(self, 'locked_output_size'):
            locked_h, locked_w = self.locked_output_size
            current_h, current_w = cropped.shape[:2]
            
            if current_h != locked_h or current_w != locked_w:
                # Create canvas of locked size
                fixed_result = np.zeros((locked_h, locked_w, 3), dtype=np.uint8)
                # Copy what fits
                copy_h = min(current_h, locked_h)
                copy_w = min(current_w, locked_w)
                fixed_result[:copy_h, :copy_w] = cropped[:copy_h, :copy_w]
                return fixed_result
        elif self.moving_camera and not hasattr(self, 'locked_output_size'):
            # Lock the output size on first successful frame
            self.locked_output_size = (cropped.shape[0], cropped.shape[1])
            logger.info(f"Locked output size: {self.locked_output_size[1]}x{self.locked_output_size[0]}")
        
        return cropped

    def fillFromSourceImages(self, result, imageA, imageB, H):
        """Fill black regions using source images."""
        h, w_left = imageB.shape[:2]
        h_right, w_right = imageA.shape[:2]
        result_h, result_w = result.shape[:2]
        
        if self.use_cuda:
            try:
                self.gpu_temp.upload(result, self.stream)
                gpu_gray = cv2.cuda.cvtColor(self.gpu_temp, cv2.COLOR_BGR2GRAY, stream=self.stream)
                self.stream.waitForCompletion()
                gray = gpu_gray.download()
            except cv2.error:
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        black_mask = gray < self.BLACK_PIXEL_THRESHOLD
        
        if not np.any(black_mask):
            return result
        
        # Fill left side from imageB
        left_region_mask = black_mask[:h, :w_left]
        if np.any(left_region_mask):
            if self.use_cuda:
                try:
                    self.gpu_temp.upload(imageB, self.stream)
                    gpu_left_gray = cv2.cuda.cvtColor(self.gpu_temp, cv2.COLOR_BGR2GRAY, stream=self.stream)
                    self.stream.waitForCompletion()
                    left_gray = gpu_left_gray.download()
                except cv2.error:
                    left_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            else:
                left_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            
            valid_fill = left_region_mask & (left_gray > self.BLACK_PIXEL_THRESHOLD)
            if np.any(valid_fill):
                valid_fill_3 = np.dstack([valid_fill] * 3)
                result[:h, :w_left][valid_fill_3] = imageB[valid_fill_3]
        
        # Fill right side from imageA (inverse warp) - vectorized
        if self.use_cuda:
            try:
                self.gpu_temp.upload(result, self.stream)
                gpu_gray = cv2.cuda.cvtColor(self.gpu_temp, cv2.COLOR_BGR2GRAY, stream=self.stream)
                self.stream.waitForCompletion()
                gray = gpu_gray.download()
            except cv2.error:
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        black_mask = gray < self.BLACK_PIXEL_THRESHOLD
        
        if np.any(black_mask):
            try:
                H_inv = np.linalg.inv(H)
                black_coords = np.where(black_mask)
                if len(black_coords[0]) > 0:
                    y_coords = black_coords[0]
                    x_coords = black_coords[1]
                    
                    pts = np.float32(np.column_stack([x_coords, y_coords])).reshape(-1, 1, 2)
                    pts_transformed = cv2.perspectiveTransform(pts, H_inv)
                    pts_transformed = pts_transformed.reshape(-1, 2)
                    
                    # Vectorized coordinate calculation
                    src_x = np.round(pts_transformed[:, 0]).astype(np.int32)
                    src_y = np.round(pts_transformed[:, 1]).astype(np.int32)
                    
                    # Create validity mask
                    valid = (src_x >= 0) & (src_x < w_right) & (src_y >= 0) & (src_y < h_right)
                    
                    # Filter to valid indices
                    valid_indices = np.where(valid)[0]
                    if len(valid_indices) > 0:
                        valid_src_x = src_x[valid_indices]
                        valid_src_y = src_y[valid_indices]
                        valid_dst_y = y_coords[valid_indices]
                        valid_dst_x = x_coords[valid_indices]
                        
                        # Get source pixels and check brightness
                        src_pixels = imageA[valid_src_y, valid_src_x]
                        brightness = np.mean(src_pixels, axis=1)
                        bright_enough = brightness > self.BLACK_PIXEL_THRESHOLD
                        
                        # Apply only bright pixels
                        final_indices = np.where(bright_enough)[0]
                        if len(final_indices) > 0:
                            result[valid_dst_y[final_indices], valid_dst_x[final_indices]] = src_pixels[final_indices]
                            
            except np.linalg.LinAlgError:
                logger.warning("Could not invert homography for fill")
        
        return result

    def detectAndDescribe(self, image):
        """Feature detection using CPU SIFT."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (kps, features) = self.detector.detectAndCompute(gray, None)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        """Feature matching."""
        if featuresA is None or featuresB is None:
            return None
        
        if len(kpsA) < 5 or len(kpsB) < 5:
            return None

        rawMatches = self.matcher.knnMatch(featuresA, featuresB, 2)

        matches = []
        for m_n in rawMatches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < n.distance * ratio:
                    matches.append((m.trainIdx, m.queryIdx))

        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            try:
                M_affine, inliers = cv2.estimateAffinePartial2D(
                    ptsA, ptsB,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=reprojThresh,
                    confidence=0.99,
                    maxIters=2000
                )
                if M_affine is not None and inliers is not None:
                    H = np.vstack([M_affine, [0, 0, 1]])
                    status = inliers.ravel().astype(np.uint8)
                    return (matches, H, status)
            except cv2.error:
                pass

            (H, status) = cv2.findHomography(ptsA, ptsB,
                                             cv2.RANSAC, reprojThresh)
            return (matches, H, status)

        return None


# =============================================================================
# THREADED VIDEO READER CLASS
# =============================================================================

class ThreadedVideoReader:
    """Threaded video reader for parallel frame loading."""
    def __init__(self, path, use_gpu=True, nvdec_available=False, queue_size=4):
        self.path = path
        self.use_gpu = use_gpu and nvdec_available
        self.reader = None
        self.cap = None
        self.stopped = False
        self.queue = Queue(maxsize=queue_size)
        
        if self.use_gpu:
            try:
                self.reader = cv2.cudacodec.createVideoReader(path)
                logger.info(f"Using NVDEC GPU decoding for {path}")
            except Exception as e:
                logger.debug(f"NVDEC failed ({e}), using CPU")
                self.use_gpu = False
        
        if not self.use_gpu:
            self.cap = cv2.VideoCapture(path)
            logger.info(f"Using CPU decoding for {path}")
        
        # Start background thread
        self.thread = Thread(target=self._reader_thread, daemon=True)
        self.thread.start()
    
    def _reader_thread(self):
        """Background thread that reads frames into queue."""
        while not self.stopped:
            if not self.queue.full():
                if self.use_gpu:
                    ret, gpu_frame = self.reader.nextFrame()
                    if ret:
                        frame = gpu_frame.download()
                        if frame.shape[2] == 4:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        self.queue.put((True, frame))
                    else:
                        self.queue.put((False, None))
                        self.stopped = True
                else:
                    ret, frame = self.cap.read()
                    self.queue.put((ret, frame))
                    if not ret:
                        self.stopped = True
            else:
                time.sleep(0.001)  # Small delay if queue is full
    
    def read(self):
        """Get frame from queue."""
        return self.queue.get()
    
    def get(self, prop):
        if self.use_gpu:
            if prop == cv2.CAP_PROP_FPS:
                try:
                    fmt = self.reader.format()
                    return fmt.fps if hasattr(fmt, 'fps') else 30.0
                except (AttributeError, cv2.error):
                    return 30.0
            elif prop == cv2.CAP_PROP_FRAME_COUNT:
                return -1
            return 0
        else:
            return self.cap.get(prop)
    
    def isOpened(self):
        if self.use_gpu:
            return self.reader is not None
        return self.cap.isOpened()
    
    def release(self):
        self.stopped = True
        if self.cap:
            self.cap.release()


# =============================================================================
# SYSTEM CHECK FUNCTIONS
# =============================================================================

def check_system():
    """Check available GPU acceleration features."""
    logger.info("=" * 60)
    logger.info("SYSTEM CHECK")
    logger.info("=" * 60)

    # Check CUDA
    cuda_available = False
    try:
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_count > 0:
            cuda_available = True
            logger.info(f"CUDA: {cuda_count} GPU(s) available")
    except Exception:
        logger.info("CUDA: Not available")

    # Check NVDEC (GPU video decoding)
    nvdec_available = False
    try:
        # Test if cudacodec is available
        if hasattr(cv2, 'cudacodec') and hasattr(cv2.cudacodec, 'createVideoReader'):
            nvdec_available = True
            logger.info("NVDEC: Available (cv2.cudacodec)")
        else:
            logger.info("NVDEC: cv2.cudacodec not available")
    except AttributeError:
        logger.info("NVDEC: cv2.cudacodec not available")

    # Check FFMPEG with NVENC
    nvenc_available = False
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'], 
            capture_output=True, text=True
        )
        if 'h264_nvenc' in result.stdout:
            nvenc_available = True
            logger.info("FFMPEG NVENC: Available (h264_nvenc)")
    except Exception:
        logger.info("FFMPEG: Not found")

    logger.info("=" * 60)
    
    return cuda_available, nvdec_available, nvenc_available


# =============================================================================
# MAIN VIDEO STITCHING PIPELINE
# =============================================================================

def stitch_videos(left_video_path, right_video_path, output_name=None, show_preview=True, moving_camera=False):
    """
    Main function to stitch two video files into a panorama.
    
    Args:
        left_video_path: Path to left video file
        right_video_path: Path to right video file
        output_name: Optional output filename (auto-generated if None)
        show_preview: Whether to show live preview window
        moving_camera: Enable moving camera mode (recalculates homography periodically)
    
    Returns:
        Path to output file
    """
    # System check
    cuda_available, nvdec_available, nvenc_available = check_system()
    
    # Open videos with threaded readers for parallel loading
    logger.info("Opening videos (threaded)...")
    left_video = ThreadedVideoReader(left_video_path, use_gpu=cuda_available, nvdec_available=nvdec_available)
    right_video = ThreadedVideoReader(right_video_path, use_gpu=cuda_available, nvdec_available=nvdec_available)

    if not left_video.isOpened():
        raise IOError(f"Could not open {left_video_path}")
    if not right_video.isOpened():
        raise IOError(f"Could not open {right_video_path}")

    # Get video properties
    fps_left = left_video.get(cv2.CAP_PROP_FPS)
    fps_right = right_video.get(cv2.CAP_PROP_FPS)
    out_fps = min(fps_left, fps_right) if fps_left > 0 and fps_right > 0 else 30.0

    # Always get frame count using CPU VideoCapture (works even with NVDEC)
    cap_temp = cv2.VideoCapture(left_video_path)
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release()

    logger.info(f"FPS: {out_fps:.2f}, Total Frames: {total_frames}")

    # Create stitcher
    stitcher = Stitcher(use_cuda=cuda_available, moving_camera=moving_camera)

    # Output setup
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    temp_output = f"temp_stitched_{timestamp}.avi"
    
    if output_name:
        final_output = output_name
    else:
        final_output = f"stitched_output_{timestamp}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = None
    output_size = None

    frame_count = 0
    total_time = 0
    fps_history = []

    logger.info("=" * 60)
    logger.info("STITCHING (Full GPU Pipeline + Threaded I/O)")
    logger.info("=" * 60)
    logger.info("GPU Operations:")
    logger.info(f"  Video decoding: {'NVDEC GPU' if nvdec_available else 'CPU'} (threaded)")
    logger.info("  Feature detection: CPU SIFT")
    logger.info(f"  Warp/Transform: {'CUDA' if cuda_available else 'CPU'}")
    logger.info(f"  Color conversion: {'CUDA' if cuda_available else 'CPU'}")
    logger.info(f"  Gaussian blur: {'CUDA' if cuda_available else 'CPU'}")
    logger.info(f"  Encoding: {'NVENC GPU' if nvenc_available else 'CPU'}")
    logger.info(f"  Moving camera mode: {'ENABLED' if moving_camera else 'DISABLED'}")
    logger.info("-" * 60)

    while True:
        frame_start = time.perf_counter()
        
        retL, left = left_video.read()
        retR, right = right_video.read()

        if not retL or not retR:
            logger.info("Videos finished.")
            break

        # Ensure same size
        if left.shape[:2] != right.shape[:2]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]))

        # Stitch
        stitch_start = time.perf_counter()
        stitched = stitcher.stitch([left, right])
        stitch_time = time.perf_counter() - stitch_start

        if stitched is None:
            logger.warning("Stitch failed, skipping frame.")
            continue

        # Create writer on first frame
        if writer is None:
            h, w = stitched.shape[:2]
            output_size = (w, h)
            writer = cv2.VideoWriter(temp_output, fourcc, out_fps, output_size)
            if not writer.isOpened():
                raise IOError("Could not create video writer")
            logger.info(f"Output size: {w}x{h}")
            logger.info("-" * 60)

        writer.write(stitched)
        frame_count += 1

        # Performance tracking
        frame_time = time.perf_counter() - frame_start
        total_time += frame_time
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_history.append(current_fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        # Get system stats
        cpu_percent = psutil.cpu_percent()
        ram_mb = PROCESS.memory_info().rss / (1024 * 1024)
        
        # Get GPU stats
        gpu_util = 0
        gpu_mem_mb = 0
        if GPU_AVAILABLE:
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE).gpu
                gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
                gpu_mem_mb = gpu_mem_info.used / (1024 * 1024)
            except pynvml.NVMLError:
                pass

        if total_frames > 0:
            progress = frame_count / total_frames
            eta = (total_frames - frame_count) / avg_fps if avg_fps > 0 else 0
            
            # Progress bar (ASCII compatible)
            bar_width = 15
            filled = int(bar_width * progress)
            bar = "#" * filled + "-" * (bar_width - filled)
            
            if GPU_AVAILABLE:
                print(f"\r[{bar}] {progress*100:5.1f}% | "
                      f"F:{frame_count}/{total_frames} | "
                      f"FPS:{avg_fps:4.1f} | "
                      f"CPU:{cpu_percent:4.1f}% | "
                      f"RAM:{ram_mb:5.0f}M | "
                      f"GPU:{gpu_util:3.0f}% | "
                      f"VRAM:{gpu_mem_mb:5.0f}M | "
                      f"ETA:{eta:4.0f}s    ", end="", flush=True)
            else:
                print(f"\r[{bar}] {progress*100:5.1f}% | "
                      f"Frame {frame_count}/{total_frames} | "
                      f"FPS: {avg_fps:.1f} | "
                      f"CPU: {cpu_percent:4.1f}% | "
                      f"RAM: {ram_mb:6.1f}MB | "
                      f"ETA: {eta:.0f}s    ", end="", flush=True)
        else:
            if GPU_AVAILABLE:
                print(f"\rFrame {frame_count} | "
                      f"FPS: {current_fps:.1f} (avg: {avg_fps:.1f}) | "
                      f"CPU: {cpu_percent:4.1f}% | "
                      f"RAM: {ram_mb:5.0f}M | "
                      f"GPU: {gpu_util:3.0f}% | "
                      f"VRAM: {gpu_mem_mb:5.0f}M    ", end="", flush=True)
            else:
                print(f"\rFrame {frame_count} | "
                      f"FPS: {current_fps:.1f} (avg: {avg_fps:.1f}) | "
                      f"CPU: {cpu_percent:4.1f}% | "
                      f"RAM: {ram_mb:6.1f}MB | "
                      f"Stitch: {stitch_time*1000:.1f}ms    ", end="", flush=True)

        # Preview
        if show_preview:
            preview_height = 300
            
            scale_left = preview_height / left.shape[0]
            left_preview = cv2.resize(left, (int(left.shape[1] * scale_left), preview_height))
            
            scale_right = preview_height / right.shape[0]
            right_preview = cv2.resize(right, (int(right.shape[1] * scale_right), preview_height))
            
            scale_stitch = preview_height / stitched.shape[0]
            stitch_preview = cv2.resize(stitched, (int(stitched.shape[1] * scale_stitch), preview_height))
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(left_preview, "LEFT", (10, 30), font, 0.8, (0, 255, 0), 2)
            cv2.putText(right_preview, "RIGHT", (10, 30), font, 0.8, (0, 255, 0), 2)
            cv2.putText(stitch_preview, "STITCHED", (10, 30), font, 0.8, (0, 255, 0), 2)
            
            top_row = np.hstack([left_preview, right_preview])
            
            if top_row.shape[1] != stitch_preview.shape[1]:
                if top_row.shape[1] < stitch_preview.shape[1]:
                    pad_width = stitch_preview.shape[1] - top_row.shape[1]
                    top_row = np.hstack([top_row, np.zeros((preview_height, pad_width, 3), dtype=np.uint8)])
                else:
                    pad_width = top_row.shape[1] - stitch_preview.shape[1]
                    stitch_preview = np.hstack([stitch_preview, np.zeros((preview_height, pad_width, 3), dtype=np.uint8)])
            
            combined_preview = np.vstack([top_row, stitch_preview])
            
            cv2.imshow("Panorama Preview (q to quit)", combined_preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User stopped.")
                break

    # Cleanup
    left_video.release()
    right_video.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    logger.info(f"Phase 1 complete: {frame_count} frames in {total_time:.1f}s ({frame_count/total_time:.1f} FPS)")

    # Phase 2: Re-encode with NVENC
    output_file = temp_output
    if nvenc_available and os.path.exists(temp_output):
        logger.info("=" * 60)
        logger.info("ENCODING (Phase 2: FFMPEG NVENC - GPU)")
        logger.info("=" * 60)
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', temp_output,
            '-c:v', 'h264_nvenc',
            '-preset', 'p4',
            '-rc', 'vbr',
            '-cq', '23',
            '-b:v', '8M',
            '-maxrate', '12M',
            '-bufsize', '16M',
            '-profile:v', 'high',
            '-pix_fmt', 'yuv420p',
            final_output
        ]
        
        logger.info(f"Running: {' '.join(ffmpeg_cmd[:8])}...")
        
        encode_start = time.perf_counter()
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        encode_time = time.perf_counter() - encode_start
        
        if result.returncode == 0:
            logger.info(f"NVENC encoding complete in {encode_time:.1f}s")
            os.remove(temp_output)
            output_file = final_output
        else:
            logger.error("NVENC failed, keeping MJPG output")
            logger.error(f"Error: {result.stderr[-500:] if result.stderr else 'Unknown'}")
            output_file = temp_output
    else:
        logger.info("Skipping NVENC (not available or no temp file)")
        output_file = temp_output

    # Final stats
    peak_ram_mb = PROCESS.memory_info().rss / (1024 * 1024)
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Frames processed: {frame_count}")
    logger.info(f"Total time: {total_time:.1f}s ({frame_count/total_time:.1f} FPS)")
    logger.info(f"Peak RAM usage: {peak_ram_mb:.1f} MB")
    
    # GPU memory stats
    if GPU_AVAILABLE:
        try:
            gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
            gpu_name = pynvml.nvmlDeviceGetName(GPU_HANDLE)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"GPU VRAM usage: {gpu_mem_info.used / (1024 * 1024):.1f} MB / {gpu_mem_info.total / (1024 * 1024):.1f} MB")
        except pynvml.NVMLError:
            pass
    
    logger.info(f"Output: {output_file}")

    if os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"File size: {size_mb:.1f} MB")

    logger.info("GPU Acceleration Summary:")
    logger.info(f"  Decoding:  {'NVDEC GPU' if nvdec_available else 'CPU'}")
    logger.info(f"  Stitching: {'CUDA GPU' if cuda_available else 'CPU'}")
    logger.info(f"  Encoding:  {'NVENC GPU' if nvenc_available and output_file.endswith('.mp4') else 'CPU'}")
    logger.info("=" * 60)
    
    return output_file


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # --- Video files (EDIT THESE) ---
    LEFT_VIDEO = "video5.mp4"
    RIGHT_VIDEO = "video6.mp4"
    
    stitch_videos(
        left_video_path=LEFT_VIDEO,
        right_video_path=RIGHT_VIDEO,
        output_name=None,           # Auto-generates timestamped name, or set like "output.mp4"
        show_preview=True,
        moving_camera=False         # Set to True for handheld/moving camera footage
    )