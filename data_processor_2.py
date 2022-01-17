import os
import cv2

filenames = os.listdir("./DIV2K_train_HR/")
scale = 3

crop_size_lr = 48
crop_size_hr = 48

dataset = "dataset_cubic.txt"

with open(dataset, "w") as f:
    pass

with open(dataset, "a") as f:
    for p in filenames:
        image_file = "./DIV2K_train_HR/" + p
        image_decoded = cv2.imread(image_file)
        cropped = image_decoded[0:(image_decoded.shape[0] - (image_decoded.shape[0] % scale)),
                    0:(image_decoded.shape[1] - (image_decoded.shape[1] % scale)), :]
        lr = cv2.resize(cropped, (int(cropped.shape[1] / scale), int(cropped.shape[0] / scale)),
                        interpolation=cv2.INTER_AREA)
        upscaled = cv2.resize(lr, (lr.shape[1] * scale, lr.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

        numx = int(lr.shape[0] / crop_size_lr)
        numy = int(lr.shape[1] / crop_size_lr)
        for i in range(0, numx, 4):
            startx = i * crop_size_lr
            endx = (i * crop_size_lr) + crop_size_lr
            for j in range(0, numy, 2):
                starty = j * crop_size_lr
                endy = (j * crop_size_lr) + crop_size_lr

                crop_lr = upscaled[startx:endx, starty:endy, :]
                crop_hr = cropped[startx:endx, starty:endy, :]

                image_file_split = p.split('.')
                image_name = image_file_split[0]
                image_extension = image_file_split[1]
                cv2.imwrite("./DIV2K_processed_cubic/" + image_name + "-" + str(i) + "-" + str(j) + "_X_" + "." + image_extension, crop_lr)
                cv2.imwrite("./DIV2K_processed_cubic/" + image_name + "-" + str(i) + "-" + str(j) + "_y_" + "." + image_extension, crop_hr)
                f.write(image_name + "-" + str(i) + "-" + str(j) + "\n")