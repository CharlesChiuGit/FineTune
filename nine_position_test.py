from finetune import fine_tuning
from renderer.io import load_moon_mesh
from renderer import build_renderer, render_single_image
from helper import camera_direction
from config import DEVICE, OBJ_filename, EXP_NAME, MOON_RADIUS
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import xlsxwriter
import random

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
plt.style.use('dark_background')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')


def got_nine_position(_init_pos):
    potential_position = []
    u_x, u_y, u_z = camera_direction(_init_pos[0], _init_pos[1], _init_pos[2],
                                     _init_pos[3], _init_pos[4], _init_pos[5])
    potential_position.append(_init_pos + [u_x, u_y, u_z])
    pos_1_extra = [0, -10, 0, 0, 0, 0]
    pos_2_extra = [0, -5, 15, 0, 0, 0]
    pos_3_extra = [0, 0, 30, 0, 0, 0]
    pos_4_extra = [0, 5, 15, 0, 0, 0]
    pos_5_extra = [0, 10, 0, 0, 0, 0]
    pos_6_extra = [0, 5, -15, 0, 0, 0]
    pos_7_extra = [0, 0, -30, 0, 0, 0]
    pos_8_extra = [0, -5, -15, 0, 0, 0]
    pos_extra_list = [pos_1_extra, pos_2_extra, pos_3_extra,
                      pos_4_extra, pos_5_extra, pos_6_extra,
                      pos_7_extra, pos_8_extra]
    for i in range(8):
        _pos = list(np.array(_init_pos) + np.array(pos_extra_list[i]))
        u_x, u_y, u_z = camera_direction(_pos[0], _pos[1], _pos[2],
                                         _pos[3], _pos[4], _pos[5])
        potential_position.append(_pos + [u_x, u_y, u_z])

    return potential_position


def get_spherical_errors(error1, error2):
    raw_error = np.array(error1).squeeze() - np.array(error2).squeeze()
    new_error = np.array([raw_error[0] * 1000, abs(raw_error[1]) % 180, abs(raw_error[2]) % 360])

    return new_error


def get_elev_error():
    elev_error_degree = 7
    elev_error = random.uniform(-elev_error_degree, elev_error_degree)
    # if target_dist == 1:
    #     elev_error = 0.5
    # elif target_dist == 5:
    #     elev_error = 3
    # elif target_dist == 10:
    #     elev_error = 8

    return elev_error


def get_azim_error():
    azim_error_degree = 16
    azim_error = random.uniform(-azim_error_degree, azim_error_degree)
    # if target_dist == 1:
    #     azim_error = 1
    # elif target_dist == 5:
    #     azim_error = 6
    # elif target_dist == 10:
    #     azim_error = 16

    return azim_error


def get_cam_pos(target_dist, init_dist, elev, azim, pos_type="target"):  # get spherical pos
    if pos_type == "target":
        cam_pos = [(target_dist / 1000) + MOON_RADIUS, elev, azim, 0, 0, 0]
    else:
        cam_pos = [(init_dist / 1000) + MOON_RADIUS,
                   get_elev_error() + elev,
                   get_azim_error() + azim, 0, 0, 0]

    return cam_pos


if __name__ == '__main__':
    logging.info('{}_Test'.format(EXP_NAME))
    wb = xlsxwriter.Workbook('./{}_Test.xlsx'.format(EXP_NAME))
    ws = wb.add_worksheet('{}_Test'.format(EXP_NAME))
    ws.set_column("A:Z", 15)
    bold = wb.add_format({'bold': 1})
    ws.write('A1', 'P&T&G index', bold)
    ws.write('B1', 'Target Cam Pos', bold)
    ws.write('E1', 'Init Cam Pos', bold)
    ws.write('H1', 'Best Cam Pos', bold)
    ws.write('K1', 'Error before F.T.', bold)
    ws.write('N1', 'Error after F.T.', bold)
    ws.write('Q1', 'Is Dist better?', bold)
    ws.write('R1', 'Is Elev better?', bold)
    ws.write('S1', 'Is Azim better?', bold)

    mesh = load_moon_mesh(OBJ_filename)
    image_size = 800
    renderer = build_renderer(image_size)

    dist_pairs = np.array([[0.7, 1], [0.8, 1], [1, 1], [1.2, 1], [1.3, 1],
                            [4, 5], [4.5, 5], [5, 5], [5.5, 5], [6, 5],
                            [9, 10], [9.5, 10], [10, 10], [10.5, 10], [11, 10]])
    target_elevs = np.array(range(0, 1)) * (180 / 1)
    target_azims = np.array(range(0, 1)) * (360 / 1)

    p = 1
    row, col = 1, 0
    for target_azim in target_azims:
        t = 1
        for target_elev in target_elevs:
            g = 1
            for dist_pair in dist_pairs:
                # target
                # target_pos = [1.74, 0, 0, 0, 0, 0]
                target_pos = get_cam_pos(dist_pair[1], dist_pair[0], target_elev, target_azim, "target")
                u_x, u_y, u_z = camera_direction(target_pos[0], target_pos[1], target_pos[2],
                                                 target_pos[3], target_pos[4], target_pos[5])
                target_image = render_single_image(mesh, renderer, (target_pos + [u_x, u_y, u_z]))  # (size, size, 3)

                # init_pos = [1.74, 0, 0, 0, 0, 0]
                init_pos = get_cam_pos(dist_pair[1], 20.0, target_elev, target_azim, "init")
                nine_init_pos = got_nine_position(init_pos)

                for i in range(9):
                    logging.info("Azim: {}, Elev :{}, Dist: {}, Potential: {}".format(p, t, g, i))
                    ws_data = []
                    # Fine Tune
                    best_pos = fine_tuning(target_image, nine_init_pos[i])
                    print("Done")

                    # Write & Plot
                    test_index = "{}_{}_{}_{}".format(p, t, g, i)
                    ws_data.append(test_index)
                    ws_data.extend(target_pos)
                    ws_data.extend(init_pos)
                    ws_data.extend(best_pos)
                    before_error = get_spherical_errors(init_pos[:3], target_pos[:3])
                    ws_data.extend(before_error)
                    after_error = get_spherical_errors(best_pos[:3], target_pos[:3])
                    ws_data.extend(after_error)
                    np.set_printoptions(precision=4)
                    print("Before Error", before_error)
                    print("After  Error", after_error)
                    compare_error = [(abs(after_error[0]) < abs(before_error[0])),
                                    (after_error[1] < before_error[1]),
                                    (after_error[2] < before_error[2])]
                    print(compare_error)
                    ws_data.extend(compare_error)
                    ws.write_row(row, col, ws_data)
                    row += 1
                g += 1
            t += 1
        p += 1
    wb.close()
