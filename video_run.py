import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os
import itertools
from main import *

class LaneParameters:
    def __init__(self):
        self.thresholds = {'h':(20,25), 'l':(200,255)}
        self.kernel_size = 3
        self.height = 720
        self.width = 1280
        self.dst_polygon_margin = 350
        self.x_l_line_center = self.dst_polygon_margin
        self.x_r_line_center = self.width - self.dst_polygon_margin
        self.stripe_height_px = 20 #Number of pixels for one stripe
        self.stripe_width_px = None
        self.stripe_width_px_reduced = self.width / 4
        self.x_m_per_px = 3.7 / 700
        self.y_m_per_px = 30.0 / 720
        self.memory = 10
        self.curv_avg_len = 0
        self.stats_l = LineStats()
        self.stats_r = LineStats()
        self.curv_min_lim = 150
        self.curv_max_mva_delta = 1000
        self.max_dropped_lines_before_reset = 10

class LineStats:
    def __init__(self):
        self.x = []
        self.y = []
        self.curv_tot = []
        self.curv_loc = []
        self.curv_avg = None
        self.total_line_drop = 0
        self.consec_line_drop = 0
        self.self = 0
    
def get_parameters():
    params = LaneParameters()  
    return params
    

def process_image(img_rgb, params, prev_line_l, prev_line_r, debug_plot=False, debug_save=False, output_dir=""):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # return the final output (image where lines are drawn on lanes)

    # Correct Camera Distortion
    img_undist = correct_img_distortion(img_rgb, params.mtx, params.dist)

    # Do Binary tresholding
    img_bin = do_binary_thresholding(img_undist, params.thresholds, params.kernel_size)

    # Transform the image perfective to bird-eye view
    img_birdeye = transform_img_perspective(img_bin, params.M_org_bird, params.width, params.height)

    # Find the coordinates of right and left lines
    (line_l, line_r) = find_lines(img_birdeye, params.stripe_height_px, params.x_l_line_center, params.x_r_line_center, params.stripe_width_px, params.stripe_width_px_reduced)

    # Convert those coordinates to metric dimensions.
    (line_l.x_m, line_l.y_m) = calc_line_coord_m(line_l)
    (line_r.x_m, line_r.y_m) = calc_line_coord_m(line_r)

    # Do curve fitting
    y_step_size = 5.0
    y_interp_m = np.arange(0.0, params.height + y_step_size, y_step_size) * params.y_m_per_px
    (coef_l_m, x_l_interp_m) = fit_line(line_l.y_m, line_l.x_m, y_interp_m)
    (coef_r_m, x_r_interp_m) = fit_line(line_r.y_m, line_r.x_m, y_interp_m)

    # Extract position of line centers
    line_l.x_center, line_r.x_center = None, None
    if len(x_l_interp_m) > 0:
        line_l.x_center = x_l_interp_m[-1] / params.x_m_per_px
    if len(x_r_interp_m) > 0:
        line_r.x_center = x_r_interp_m[-1] / params.x_m_per_px

    # Store values
    line_l.coef_m, line_l.x_m_interp, line_l.y_m_interp = coef_l_m, x_l_interp_m, y_interp_m
    line_r.coef_m, line_r.x_m_interp, line_r.y_m_interp = coef_r_m, x_r_interp_m, y_interp_m
    line_l.x_interp, line_l.y_interp = x_l_interp_m / params.x_m_per_px, y_interp_m / params.y_m_per_px
    line_r.x_interp, line_r.y_interp = x_r_interp_m / params.x_m_per_px, y_interp_m / params.y_m_per_px

    # Calculate line curvature, for the y coordinate the closest to the car
    line_l.curv_m = calc_line_curvature_m(line_l, params.height * params.y_m_per_px)
    line_r.curv_m = calc_line_curvature_m(line_r, params.height * params.y_m_per_px)

    # Verify and correct line detection if anomalies detected
    if prev_line_l != None and prev_line_r != None:
        (line_l, line_r) = verify_correct_smooth_line_detection(line_l, line_r, prev_line_l, prev_line_r, params)

    # Calculate statistics to overlay on image
    params.curv_avg_m = np.mean([line_l.curv_m, line_r.curv_m])
    params.offset_m = calc_offset_from_center(line_l, line_r, params.width, params.x_m_per_px)

    # Convert back to original view
    img_poly = convert_lines_to_original_view(line_l, line_r, params.width, params.height, params.M_bird_org)

    # Add Text to image
    font = cv2.FONT_HERSHEY_TRIPLEX 
    text_offset = 'Offset: {:.2f} m'.format(params.offset_m)
    text_curv = 'Curvature: {:.1f} m'.format(params.curv_avg_m)
    cv2.putText(img_poly,text_offset,(10,40), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(img_poly,text_curv,(10,90), font, 1,(255,255,255),2,cv2.LINE_AA)    

    # Combine original image with polygon
    img_res = cv2.addWeighted(img_undist, 1, img_poly, 0.3, 0)

    if debug_plot:
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))
        ax1.set_title('Undistorted')
        ax1.imshow(img_undist)

        ax2.set_title('Binary Thresholding')
        ax2.imshow(img_bin, cmap='gray')

        ax3.set_title('Bird-eye view + Line Detection')
        ax3.imshow(img_birdeye, cmap='gray')
        ax3.plot(line_l.x, line_l.y, linestyle='None', marker='.', color='r', markersize=15)
        ax3.plot(line_r.x, line_r.y, linestyle='None', marker='.', color='r', markersize=15)
        ax3.plot(line_l.x_interp, line_l.y_interp, linestyle='-', linewidth=5, marker='None', color='g')
        ax3.plot(line_r.x_interp, line_r.y_interp, linestyle='-', linewidth=5, marker='None', color='g')

        ax4.set_title('Result: Curv. L: {:.2f}m, Curv. R: {:.2f}'.format(line_l.curv_m, line_r.curv_m))
        ax4.imshow(img_res)
        plt.show() 

    if debug_save:
        img_bin_scl = img_bin * 255
        output_file = output_dir + 'bin_frame_{}.jpg'.format(idx)
        cv2.imwrite(output_file, img_bin_scl)

        img_beye_scl = img_birdeye * 255
        output_file = output_dir + 'top_frame_{}.jpg'.format(idx)
        cv2.imwrite(output_file, img_beye_scl)

        output_file = output_dir + 'res_frame_{}.jpg'.format(idx)
        cv2.imwrite(output_file, cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR))

    return (img_res, line_l, line_r)

def get_max_curv_avg_delta(curv_avg):
    if curv_avg < 500:
        return 500
    elif curv_avg < 1000:
        return 1000
    elif curv_avg < 3000:
        return 3000
    elif curv_avg < 5000:
        return 5000
    elif curv_avg < 10000:
        return 10000
    else:
        return 50000

def correct_line(line, line_prev, label, params, stats):

    out_line = line

    # First, make sure we have detected both lines.
    # If not, we will use the previously detected line.
    bad_line = False
    nb_pts = len(line.y)
    if nb_pts == 0:
        out_line = line_prev
        stats.total_line_drop += 1
        stats.consec_line_drop += 1
        bad_line = True
        print('Warning: Found empty line ({})'.format(label))

    if bad_line:
        return (out_line, stats)

    # Make sure the curvature is withing the limits       
    if line.curv_m < params.curv_min_lim:
        print('Warning: Found dangerous Curvature at {:.2} ({})'.format(line.curv_m, label))
        out_line = line_prev
        stats.total_line_drop += 1
        stats.consec_line_drop += 1
        bad_line = True

    if bad_line:
        return (out_line, stats)

    # Check the curvature compared to the moving average
    curv_max_mva_delta = get_max_curv_avg_delta(stats.curv_avg)
    if stats.curv_avg_len > 5 and abs(line.curv_m - stats.curv_avg) > curv_max_mva_delta:
        print('Warning: Curvature too different from moving average - Curv: {:.2f}, Avg: {:.2f} ({})'.format(line.curv_m, stats.curv_avg, label))
        out_line = line_prev
        stats.total_line_drop += 1
        stats.consec_line_drop += 1
        bad_line = True

    if bad_line:
        return (out_line, stats)

    # Smooth the curve if we are within the limits
    # We use the points from the former N frames, combined together for Curve fitting

    # Reset Line Drops (If we got here, that's because it's a good line)
    stats.consec_line_drop = 0

    # Do curve fitting
    y_step_size = 5.0
    y_interp_m = np.arange(0.0, params.height + y_step_size, y_step_size) * params.y_m_per_px
    x = list(itertools.chain.from_iterable(stats.x)) + list(line.x_m)
    y = list(itertools.chain.from_iterable(stats.y)) + list(line.y_m)
    (coef_m, x_interp_m) = fit_line(y, x, y_interp_m)

    # Extract position of line center
    out_line.x_center = None
    if len(x_interp_m) > 0:
        out_line.x_center = x_interp_m[-1] / params.x_m_per_px

    # Store values
    out_line.coef_m, out_line.x_m_interp, out_line.y_m_interp = coef_m, x_interp_m, y_interp_m
    out_line.x_interp, out_line.y_interp = x_interp_m / params.x_m_per_px, y_interp_m / params.y_m_per_px

    # Calculate line curvature, for the y coordinate the closest to the car
    out_line.curv_m = calc_line_curvature_m(out_line, params.height * params.y_m_per_px)

    return (out_line, stats)


def verify_correct_smooth_line_detection(line_l, line_r, line_l_prev, line_r_prev, params):
    """Here we detect anomalies in the line detection and try to correct for them.
    We detect anomalies by comparing the current lane detection with the previous one.
    Such anomalies could be:
    - Line center too far away from previously detected line center
    - Curvature seems out of bounds
    - Curvature of the 2 lines are very different from each other
    - Very few points were detected on the line
    
    If such anomalies are detected, we correct them by using the lane with the best detection."""

    (out_line_l, params.stats_l) = correct_line(line_l, line_l_prev, 'Left', params, params.stats_l)
    (out_line_r, params.stats_r) = correct_line(line_r, line_r_prev, 'Right', params, params.stats_r)

    return (out_line_l, out_line_r)

def clip_transform(clip, params):
    def detect_lane(image):
        return process_image(image, params)
    return clip.fl_image(detect_lane)


def calc_stats(line, p, stats):

    # First check how many frames we dropped in a row. If too many were dropped, it's preferable to reset the moving average to 0.
    if stats.consec_line_drop > p.max_dropped_lines_before_reset:
        stats.x, stats.y = [], []
        stats.curv_loc = []

    # Store Detected Points. (Keep only a limited amount of frames)
    stats.x.append(list(line.x_m))
    stats.y.append(list(line.y_m))
    stats.x = stats.x[-p.memory:]
    stats.y = stats.y[-p.memory:]

    # Store line curvature (For plotting at the end)
    stats.curv_tot.append(line.curv_m)

    # Calculate the Moving average of the curve over a limited amount of frames
    stats.curv_loc.append(line.curv_m)
    stats.curv_loc = stats.curv_loc[-p.memory:]
    stats.curv_avg = np.mean(stats.curv_loc)
    stats.curv_avg_len = len(stats.curv_loc)

    return stats


if __name__ == "__main__":

    from moviepy.editor import VideoFileClip, ImageSequenceClip
    import time

    # Define input/output files
    output_dir = 'output_images/video/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    input_video = 'project_video.mp4'
    output_video = output_dir + 'output_video.mp4'

    # Get parameters
    p = get_parameters()
    origin_stripe_width = p.width/2
    p.stripe_width_px = origin_stripe_width
    p.stripe_width_px_reduced = origin_stripe_width/2

    # Calibrate Camera
    (p.mtx, p.dist, rvecs, tvecs) = calibrate_camera()

    # Get source and destination polygons (For Perspective Transform)
    (src, dst) = get_src_dst_polygons(p.height, p.width, p.dst_polygon_margin)

    # Calculate Transform Matrices
    p.M_org_bird = get_transform_matrix(src, dst)
    p.M_bird_org = get_transform_matrix(dst, src)

    # Process Frame by frame
    in_clip = VideoFileClip(input_video)
    output_dir = 'output_images/video/' + time.strftime("%Y%m%d_%H%M%S") + '/'
    os.mkdir(output_dir)

    line_l = None
    line_r = None
    out_images = []

    for idx, frame in enumerate(in_clip.iter_frames()):

        # Save Input Frame for debugging
        output_file = 'output_images/video/project_video_images/frame_{}.jpg'.format(idx)
        cv2.imwrite(output_file, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Run processing on output frame
        print("\n--------------------------------")
        print("Processing Frame {}".format(idx))
        prev_line_l, prev_line_r = line_l, line_r
        (outframe, line_l, line_r) = process_image(frame, p, prev_line_l, prev_line_r, False, False, output_dir)

        # Set start of line detection for next frame (if lines were detected)
        if line_l.x_center != None:
            p.x_l_line_center = line_l.x_center
        if line_r.x_center != None:
            p.x_r_line_center = line_r.x_center

        # Reduce stripe width if lines were detected
        if line_l.x_center != None and line_r.x_center != None:
            p.stripe_width_px = origin_stripe_width/2
        else:
            p.stripe_width_px = origin_stripe_width

        # Calculate Statistics
        p.stats_l = calc_stats(line_l, p, p.stats_l)
        p.stats_r = calc_stats(line_r, p, p.stats_r)
        print("Curvature L:{:.2f} (avg: {:.2f}), R:{:.2f} (avg: {:.2f})".format(line_l.curv_m, p.stats_l.curv_avg, line_r.curv_m, p.stats_r.curv_avg))
        print("Consecutive Line Dropped L: {}, R: {}".format(p.stats_l.consec_line_drop, p.stats_r.consec_line_drop))
        
        #Save Output frame
        output_file = output_dir + 'frame_{}.jpg'.format(idx)
        cv2.imwrite(output_file, cv2.cvtColor(outframe, cv2.COLOR_RGB2BGR))

        # Store output file
        out_images.append(output_file)

    # Make a clip
    out_clip = ImageSequenceClip(out_images, fps=24)
    out_clip.write_videofile(output_video, audio=False)

    # Show statistics
    print('\n-------------------------------')
    print('Statistics')
    print("Number of Rejected Lines: L: {} ({:.2f}), R: {} ({:.2f})".format(p.stats_l.total_line_drop, 100 * p.stats_l.total_line_drop / idx, p.stats_r.total_line_drop, 100 * p.stats_r.total_line_drop / idx))
    # Plot Line curvature
    f, ax1 = plt.subplots(1, 1, figsize=(16,8))
    ax1.set_title('Line Curvature')
    ax1.plot(np.arange(0, len(p.stats_l.curv_tot)), p.stats_l.curv_tot)
    ax1.plot(np.arange(0, len(p.stats_r.curv_tot)), p.stats_r.curv_tot)
    plt.show() 

