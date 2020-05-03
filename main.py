from finetune import fine_tuning


if __name__ == '__main__':

    target_image = "img_path/XXXX.png"
    init_pos = [1.74, 0, 0, 0, 0, 0, 0, 1, 0]  # [c_dist, c_elev, c_azim, p_x, p_y, p_z, u_x, u_y, u_z]

    # Fine Tune
    best_pos_sphe = fine_tuning(target_image, init_pos)
    print(best_pos_sphe)