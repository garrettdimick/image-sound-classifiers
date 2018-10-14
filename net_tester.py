## get the bee test data

ann_test_predata = []
convnet_test_predata = []
f = glob.glob("/Users/garrettdimick/Google Drive/Fall 2018/CS5600-AI/Project_01/BEE2Set/bee_test/*/*.png")
for file in f:
    img = cv2.imread(file)
    convnet_test_data.append(img/float(255))
    ## Grayscale the images for the ANN
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ann_test_data.append(gray_img/float(255))

## get the NO bee test data
