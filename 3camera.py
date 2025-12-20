#!/usr/bin/env python3
"""
GPU-Accelerated Video Panorama Stitcher - 3 CAMERA VERSION
Stitches LEFT + CENTER + RIGHT cameras into a wide panorama.

Based on moving.py with sequential stitching approach.
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
    OVERLAP_THRESHOLD_RATIO = 0.15
    FALLBACK_THRESHOLD_RATIO = 0.05
    BLEND_WIDTH = 100
    BLACK_PIXEL_THRESHOLD = 10
    PARALLAX_DIFF_THRESHOLD = 15
    VALID_PIXEL_MIN = 50
    VALID_PIXEL_MAX = 220
    MIN_VALID_PIXELS = 500
    PERSPECTIVE_THRESHOLD = 0.005
    SCALE_MAX = 1.5
    SCALE_MIN = 0.67
    LUMINANCE_DARK_THRESHOLD = 60
    LUMINANCE_BRIGHT_THRESHOLD = 200
    DILATION_KERNEL_SIZE = 40
    
    # Moving camera parameters
    KEYFRAME_INTERVAL = 30
    HOMOGRAPHY_CHANGE_THRESHOLD = 0.1
    HOMOGRAPHY_BUFFER_SIZE = 5
    MIN_MATCH_COUNT = 50
    FEATURE_COUNT = 8000
    
    def __init__(self, use_cuda=True, moving_camera=False, name=""):
        self.cachedH = None
        self.blend_start = None
        self.blend_end = None
        self.output_width = None
        self.output_height = None
        self.crop_top = None
        self.crop_bottom = None
        self.name = name  # For logging which stitcher
        
        # Moving camera support
        self.moving_camera = moving_camera
        self.frame_count = 0
        self.homography_buffer = []
        self.last_match_count = 0
        
        # Check CUDA availability
        self.use_cuda = use_cuda and self._check_cuda()
        
        if self.use_cuda:
            self._init_cuda()
        else:
            logger.info(f"[{name}] Running on CPU")
        
        # Feature detector
        n_features = self.FEATURE_COUNT if moving_camera else 5000
        self.detector = cv2.SIFT_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.use_gpu_features = False
        logger.info(f"[{name}] Using CPU SIFT (nfeatures={n_features})")
        
        if moving_camera:
            logger.info(f"[{name}] Moving camera mode enabled")

    def _check_cuda(self):
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
        logger.info(f"[{self.name}] CUDA enabled!")
        
        self.gpu_frame = cv2.cuda_GpuMat()
        self.gpu_frameB = cv2.cuda_GpuMat()
        self.gpu_warped = cv2.cuda_GpuMat()
        self.gpu_gray = cv2.cuda_GpuMat()
        self.gpu_grayB = cv2.cuda_GpuMat()
        self.gpu_result = cv2.cuda_GpuMat()
        self.gpu_temp = cv2.cuda_GpuMat()
        self.gpu_blend_left = cv2.cuda_GpuMat()
        self.gpu_blend_right = cv2.cuda_GpuMat()
        
        self.stream = cv2.cuda_Stream()
        
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

    def _should_recalculate_homography(self):
        if not self.moving_camera:
            return self.cachedH is None
        
        if self.cachedH is None:
            return True
        
        if self.frame_count % self.KEYFRAME_INTERVAL == 0:
            return True
        
        if self.last_match_count < self.MIN_MATCH_COUNT:
            return True
        
        return False
    
    def _smooth_homography(self, H):
        self.homography_buffer.append(H.copy())
        
        if len(self.homography_buffer) > self.HOMOGRAPHY_BUFFER_SIZE:
            self.homography_buffer.pop(0)
        
        if len(self.homography_buffer) == 1:
            return H
        
        weights = np.array([i + 1 for i in range(len(self.homography_buffer))], dtype=np.float32)
        weights /= weights.sum()
        
        smoothed_H = np.zeros_like(H, dtype=np.float64)
        for i, h in enumerate(self.homography_buffer):
            smoothed_H += weights[i] * h
        
        smoothed_H /= smoothed_H[2, 2]
        
        return smoothed_H.astype(np.float32)
    
    def _homography_changed_significantly(self, new_H):
        if self.cachedH is None:
            return True
        
        diff = np.linalg.norm(new_H - self.cachedH, 'fro')
        return diff > self.HOMOGRAPHY_CHANGE_THRESHOLD

    def detectAndDescribe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (kps, features) = self.detector.detectAndCompute(gray, None)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        if featuresA is None or featuresB is None:
            return None
        if len(featuresA) == 0 or len(featuresB) == 0:
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

            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return (matches, H, status)

        return None

    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
        (imageB, imageA) = images  # left = B, right = A
        
        self.frame_count += 1

        if self._should_recalculate_homography():
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)

            M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
            if M is None:
                logger.warning(f"[{self.name}] Not enough matches")
                if self.moving_camera and self.cachedH is not None:
                    return self.applyWarp(imageA, imageB, self.cachedH)
                return None

            (matches, H, status) = M
            self.last_match_count = len(matches)
            
            if H is None:
                if self.moving_camera and self.cachedH is not None:
                    return self.applyWarp(imageA, imageB, self.cachedH)
                return None

            H = self._constrainHomography(H, imageA.shape, imageB.shape)
            
            if self.moving_camera:
                H = self._smooth_homography(H)
                
                if self._homography_changed_significantly(H):
                    self.blend_start = None
                    self.blend_end = None
                    logger.debug(f"[{self.name}] Homography updated at frame {self.frame_count}")
            
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
        h, w = imageB.shape[:2]
        
        corners = np.float32([[0, 0], [imageA.shape[1], 0], 
                              [imageA.shape[1], imageA.shape[0]], [0, imageA.shape[0]]])
        warped_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H)
        max_x = int(np.max(warped_corners[:, 0, 0]))
        canvas_width = min(max_x + 50, imageA.shape[1] + imageB.shape[1])

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
                
                gpu_warped_gray = cv2.cuda.cvtColor(gpu_warped, cv2.COLOR_BGR2GRAY, stream=self.stream)
                
                self.stream.waitForCompletion()
                warped = gpu_warped.download()
                warped_gray = gpu_warped_gray.download()
            except Exception as e:
                warped = cv2.warpPerspective(imageA, H, (canvas_width, h),
                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            warped = cv2.warpPerspective(imageA, H, (canvas_width, h),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

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
            
            actual_blend_width = self.blend_end - self.blend_start
            if actual_blend_width > 0:
                mask_1d = np.linspace(0, 1, actual_blend_width, dtype=np.float32)
                mask_1d = mask_1d * mask_1d * (3 - 2 * mask_1d)
                self.gradient_mask = np.tile(mask_1d, (h, 1))
                self.gradient_mask_3 = np.dstack([self.gradient_mask] * 3)
            
            logger.info(f"[{self.name}] Blend region: {self.blend_start} to {self.blend_end}")
            logger.info(f"[{self.name}] Output size: {self.output_width}x{self.output_height}")

        blend_start = self.blend_start
        blend_end = self.blend_end
        actual_blend_width = blend_end - blend_start

        result = warped.copy()

        # Color matching
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
                if c == 0:
                    correction = (warped_lab[:, :, c] - right_mean) * scale + left_mean
                    warped_lab[:, :, c] = warped_lab[:, :, c] + luminance_protection * 0.3 * (correction - warped_lab[:, :, c])
                else:
                    warped_lab[:, :, c] = (warped_lab[:, :, c] - right_mean) * scale + left_mean
            
            warped_lab = np.clip(warped_lab, 0, 255).astype(np.uint8)
            
            if self.use_cuda:
                try:
                    self.gpu_temp.upload(warped_lab, self.stream)
                    gpu_result = cv2.cuda.cvtColor(self.gpu_temp, cv2.COLOR_Lab2BGR, stream=self.stream)
                    self.stream.waitForCompletion()
                    result = gpu_result.download()
                except cv2.error:
                    result = cv2.cvtColor(warped_lab, cv2.COLOR_LAB2BGR)
            else:
                result = cv2.cvtColor(warped_lab, cv2.COLOR_LAB2BGR)

        # Blending
        result[:h, :blend_start] = imageB[:, :blend_start]

        if actual_blend_width > 0 and hasattr(self, 'gradient_mask_3'):
            left_blend = imageB[:, blend_start:blend_end].astype(np.float32)
            right_blend = result[:h, blend_start:blend_end].astype(np.float32)
            
            blended = left_blend * (1 - self.gradient_mask_3) + right_blend * self.gradient_mask_3
            result[:h, blend_start:blend_end] = blended.astype(np.uint8)

        return result[:, :self.output_width]
    
    def reset(self):
        """Reset stitcher state for new video pair."""
        self.cachedH = None
        self.blend_start = None
        self.blend_end = None
        self.output_width = None
        self.output_height = None
        self.frame_count = 0
        self.homography_buffer = []
        self.last_match_count = 0


# =============================================================================
# THREADED VIDEO READER CLASS
# =============================================================================

class ThreadedVideoReader:
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
        
        self.thread = Thread(target=self._reader_thread, daemon=True)
        self.thread.start()
    
    def _reader_thread(self):
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
                time.sleep(0.001)
    
    def read(self):
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
    logger.info("=" * 60)
    logger.info("SYSTEM CHECK")
    logger.info("=" * 60)

    cuda_available = False
    try:
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_count > 0:
            cuda_available = True
            logger.info(f"CUDA: {cuda_count} GPU(s) available")
    except Exception:
        logger.info("CUDA: Not available")

    nvdec_available = False
    try:
        if hasattr(cv2, 'cudacodec') and hasattr(cv2.cudacodec, 'createVideoReader'):
            nvdec_available = True
            logger.info("NVDEC: Available (cv2.cudacodec)")
        else:
            logger.info("NVDEC: cv2.cudacodec not available")
    except AttributeError:
        logger.info("NVDEC: cv2.cudacodec not available")

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
# MAIN 3-CAMERA VIDEO STITCHING PIPELINE
# =============================================================================

def stitch_videos_3cam(left_video_path, center_video_path, right_video_path, 
                       output_name=None, show_preview=True, moving_camera=False):
    """
    Main function to stitch THREE video files into a wide panorama.
    
    Stitching order: LEFT + CENTER -> intermediate, then intermediate + RIGHT -> final
    
    Args:
        left_video_path: Path to left video file (e.g., camera_1 Front-Left)
        center_video_path: Path to center video file (e.g., camera_0 Front)
        right_video_path: Path to right video file (e.g., camera_3 Front-Right)
        output_name: Optional output filename (auto-generated if None)
        show_preview: Whether to show live preview window
        moving_camera: Enable moving camera mode
    
    Returns:
        Path to output file
    """
    # System check
    cuda_available, nvdec_available, nvenc_available = check_system()
    
    # Open all three videos
    logger.info("Opening 3 videos (threaded)...")
    left_video = ThreadedVideoReader(left_video_path, use_gpu=cuda_available, nvdec_available=nvdec_available)
    center_video = ThreadedVideoReader(center_video_path, use_gpu=cuda_available, nvdec_available=nvdec_available)
    right_video = ThreadedVideoReader(right_video_path, use_gpu=cuda_available, nvdec_available=nvdec_available)

    if not left_video.isOpened():
        raise IOError(f"Could not open {left_video_path}")
    if not center_video.isOpened():
        raise IOError(f"Could not open {center_video_path}")
    if not right_video.isOpened():
        raise IOError(f"Could not open {right_video_path}")

    # Get video properties
    fps = center_video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 10.0  # Waymo default
    
    total_frames = int(center_video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Videos opened - FPS: {fps}, Frames: {total_frames if total_frames > 0 else 'unknown'}")

    # Create TWO stitchers: one for left+center, one for intermediate+right
    stitcher_LC = Stitcher(use_cuda=cuda_available, moving_camera=moving_camera, name="L+C")
    stitcher_LCR = Stitcher(use_cuda=cuda_available, moving_camera=moving_camera, name="LC+R")

    # Output setup
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if output_name:
        final_output = output_name
    else:
        final_output = f"stitched_3cam_{timestamp}.mp4"
    
    temp_output = final_output.replace('.mp4', '_temp.avi')

    logger.info("=" * 60)
    logger.info("3-CAMERA STITCHING (Phase 1: OpenCV)")
    logger.info("=" * 60)

    writer = None
    frame_count = 0
    total_time = 0
    fps_history = []
    output_size_locked = False

    while True:
        frame_start = time.perf_counter()

        # Read from all three videos
        ret_l, left = left_video.read()
        ret_c, center = center_video.read()
        ret_r, right = right_video.read()

        if not ret_l or not ret_c or not ret_r:
            logger.info("End of video(s) reached")
            break

        if left is None or center is None or right is None:
            continue

        frame_count += 1

        # === SEQUENTIAL STITCHING ===
        stitch_start = time.perf_counter()
        
        # Step 1: Stitch LEFT + CENTER
        intermediate = stitcher_LC.stitch([left, center])
        if intermediate is None:
            logger.warning(f"Frame {frame_count}: L+C stitch failed, skipping")
            continue
        
        # Step 2: Stitch INTERMEDIATE + RIGHT
        final = stitcher_LCR.stitch([intermediate, right])
        if final is None:
            logger.warning(f"Frame {frame_count}: LC+R stitch failed, skipping")
            continue
        
        stitch_time = time.perf_counter() - stitch_start

        # Lock output size on first successful frame
        if not output_size_locked:
            # Add some padding to accommodate slight variations
            output_h, output_w = final.shape[:2]
            output_w = output_w + 100  # Extra buffer for varying widths
            logger.info(f"Output size locked: {output_w}x{output_h}")
            output_size_locked = True
            
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(temp_output, fourcc, fps, (output_w, output_h))
            
            if not writer.isOpened():
                raise IOError(f"Could not open VideoWriter for {temp_output}")

        # STRICTLY enforce frame size with padding/cropping (not resize to avoid distortion)
        final_h, final_w = final.shape[:2]
        
        if final_w != output_w or final_h != output_h:
            # Create black canvas of exact output size
            canvas = np.zeros((output_h, output_w, 3), dtype=np.uint8)
            
            # Copy what fits
            copy_w = min(final_w, output_w)
            copy_h = min(final_h, output_h)
            canvas[:copy_h, :copy_w] = final[:copy_h, :copy_w]
            
            final = canvas

        writer.write(final)

        # Performance tracking
        frame_time = time.perf_counter() - frame_start
        total_time += frame_time
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_history.append(current_fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        # Stats
        cpu_percent = psutil.cpu_percent()
        ram_mb = PROCESS.memory_info().rss / (1024 * 1024)
        
        gpu_util = 0
        gpu_mem_mb = 0
        if GPU_AVAILABLE:
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE).gpu
                gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
                gpu_mem_mb = gpu_mem_info.used / (1024 * 1024)
            except:
                pass

        if total_frames > 0:
            progress = frame_count / total_frames
            eta = (total_frames - frame_count) / avg_fps if avg_fps > 0 else 0
            
            bar_width = 15
            filled = int(bar_width * progress)
            bar = "#" * filled + "-" * (bar_width - filled)
            
            print(f"\r[{bar}] {progress*100:5.1f}% | "
                  f"F:{frame_count}/{total_frames} | "
                  f"FPS:{avg_fps:4.1f} | "
                  f"CPU:{cpu_percent:4.1f}% | "
                  f"RAM:{ram_mb:5.0f}M | "
                  f"GPU:{gpu_util:3.0f}% | "
                  f"ETA:{eta:4.0f}s    ", end="", flush=True)
        else:
            print(f"\rFrame {frame_count} | "
                  f"FPS: {avg_fps:.1f} | "
                  f"CPU: {cpu_percent:4.1f}% | "
                  f"RAM: {ram_mb:5.0f}M    ", end="", flush=True)

        # Preview
        if show_preview:
            preview_height = 250
            
            # Show all 3 inputs + final
            scale_l = preview_height / left.shape[0]
            left_preview = cv2.resize(left, (int(left.shape[1] * scale_l), preview_height))
            
            scale_c = preview_height / center.shape[0]
            center_preview = cv2.resize(center, (int(center.shape[1] * scale_c), preview_height))
            
            scale_r = preview_height / right.shape[0]
            right_preview = cv2.resize(right, (int(right.shape[1] * scale_r), preview_height))
            
            scale_f = preview_height / final.shape[0]
            final_preview = cv2.resize(final, (int(final.shape[1] * scale_f), preview_height))
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(left_preview, "LEFT", (10, 30), font, 0.7, (0, 255, 0), 2)
            cv2.putText(center_preview, "CENTER", (10, 30), font, 0.7, (0, 255, 0), 2)
            cv2.putText(right_preview, "RIGHT", (10, 30), font, 0.7, (0, 255, 0), 2)
            cv2.putText(final_preview, "3-CAM PANORAMA", (10, 30), font, 0.7, (0, 255, 0), 2)
            
            # Stack inputs horizontally
            top_row = np.hstack([left_preview, center_preview, right_preview])
            
            # Match widths
            if top_row.shape[1] != final_preview.shape[1]:
                if top_row.shape[1] < final_preview.shape[1]:
                    pad = final_preview.shape[1] - top_row.shape[1]
                    top_row = np.hstack([top_row, np.zeros((preview_height, pad, 3), dtype=np.uint8)])
                else:
                    pad = top_row.shape[1] - final_preview.shape[1]
                    final_preview = np.hstack([final_preview, np.zeros((preview_height, pad, 3), dtype=np.uint8)])
            
            combined = np.vstack([top_row, final_preview])
            
            cv2.imshow("3-Camera Panorama (q to quit)", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User stopped.")
                break

    # Cleanup
    left_video.release()
    center_video.release()
    right_video.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    logger.info(f"\nPhase 1 complete: {frame_count} frames in {total_time:.1f}s ({frame_count/total_time:.1f} FPS)")

    # Phase 2: Re-encode to MP4
    output_file = temp_output
    if os.path.exists(temp_output):
        logger.info("=" * 60)
        logger.info("ENCODING (Phase 2: Converting to MP4)")
        logger.info("=" * 60)
        
        # Try NVENC first, then fallback to CPU
        if nvenc_available:
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', temp_output,
                '-c:v', 'h264_nvenc',
                '-preset', 'p4',
                '-rc', 'vbr',
                '-cq', '23',
                '-b:v', '12M',
                '-maxrate', '18M',
                '-bufsize', '24M',
                '-profile:v', 'high',
                '-pix_fmt', 'yuv420p',
                final_output
            ]
        else:
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', temp_output,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                final_output
            ]
        
        logger.info(f"Running: {' '.join(ffmpeg_cmd[:6])}...")
        
        encode_start = time.perf_counter()
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        encode_time = time.perf_counter() - encode_start
        
        if result.returncode == 0:
            logger.info(f"Encoding complete in {encode_time:.1f}s")
            os.remove(temp_output)
            output_file = final_output
        else:
            logger.error(f"Encoding failed: {result.stderr[-500:] if result.stderr else 'Unknown error'}")
            # Try CPU fallback if NVENC failed
            if nvenc_available:
                logger.info("Trying CPU encoding fallback...")
                ffmpeg_cmd_cpu = [
                    'ffmpeg', '-y',
                    '-i', temp_output,
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    final_output
                ]
                result2 = subprocess.run(ffmpeg_cmd_cpu, capture_output=True, text=True)
                if result2.returncode == 0:
                    logger.info("CPU encoding succeeded")
                    os.remove(temp_output)
                    output_file = final_output
                else:
                    logger.error("CPU encoding also failed, keeping AVI")
                    output_file = temp_output
            else:
                output_file = temp_output
    else:
        logger.info("No temp file to encode")
        output_file = temp_output

    # Final stats
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Frames processed: {frame_count}")
    logger.info(f"Total time: {total_time:.1f}s ({frame_count/total_time:.1f} FPS)")
    logger.info(f"Output: {output_file}")

    if os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"File size: {size_mb:.1f} MB")

    logger.info("=" * 60)
    
    return output_file


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # --- Video files for 3-camera panorama (EDIT THESE) ---
    # For Waymo: camera_1 (Front-Left), camera_0 (Front), camera_3 (Front-Right)
    
    LEFT_VIDEO = "camera1.mp4"      # Front-Left
    CENTER_VIDEO = "camera0.mp4"    # Front (center)
    RIGHT_VIDEO = "camera3.mp4"     # Front-Right
    
    stitch_videos_3cam(
        left_video_path=LEFT_VIDEO,
        center_video_path=CENTER_VIDEO,
        right_video_path=RIGHT_VIDEO,
        output_name=None,           # Auto-generates timestamped name
        show_preview=True,
        moving_camera=True          # Enable for Waymo moving footage
    )