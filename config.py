from os.path import join

training_dir = join("..","..","datasets","AT&T Database of Faces","faces","training")
testing_dir = join("..","..","datasets","AT&T Database of Faces","faces","testing")
training_csv = join("..","..","datasets","AT&T Database of Faces","faces","training","train_data.csv")
testing_csv = join("..","..","datasets","AT&T Database of Faces","faces","training","test_data.csv")
train_batch_size = 40
train_number_epochs = 20
learning_rate = 0.0005
weight_decay = 0.0005
img_height = 100
img_width = 100

# testing_dir = join("..","..","datasets", "tutorial","test")
# testing_csv = "test_data.csv"
