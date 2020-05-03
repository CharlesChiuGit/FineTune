from finetune import fine_tuning


if __name__ == '__main__':

    target_pos_sphe = [1.74, 5, 5]
    init_pos_sphe = [1.74, 0, 0]

    # Fine Tune
    best_pos_sphe = fine_tuning([target_pos_sphe], init_pos_sphe)
    print(best_pos_sphe)