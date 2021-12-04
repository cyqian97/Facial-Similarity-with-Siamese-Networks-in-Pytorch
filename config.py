from os.path import join

training_dir = join("..","..","datasets","AT&T Database of Faces","faces","training")
training_csv = "train_data.csv"
batch_size = 40
epochs = 20
learning_rate = 1e-6
weight_decay = 0.0005
img_height = 155
img_width = 220
alpha = 0.1
beta = 0.9

# testing_dir = join("..","..","datasets", "tutorial","test")
# testing_csv = "test_data.csv"