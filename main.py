from finetune import fine_tuning
import cv2


if __name__ == '__main__':

    target_image_path = "img_path/XXXX.png"
    target_image = cv2.imread(target_image_path)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)  # got (SIZE, SIZE, 3)
    init_pos = [1.74, 0, 0, 0, 0, 0, 0, 1, 0]  # [c_dist, c_elev, c_azim, p_x, p_y, p_z, u_x, u_y, u_z]

    # Fine Tune
    best_pos_sphe = fine_tuning(target_image, init_pos)
    print(best_pos_sphe)