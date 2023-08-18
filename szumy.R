library(caret)
library(keras)
library(imager)
library(readr)
setwd('R:/UMCS/R - dane/szumy')
a <- load.dir('train')
a1 <- load.dir('train_cleaned')
# for (i in seq(0.3, 0.7, 0.01)){
#   b <- sapply(a, function(x) threshold(x, i))
#   b <- sapply(b, function(x) imager::resize(x, size_x = 540, size_y = 420))
#   c <- sapply(a1, function(x) imager::resize(x, size_x = 540, size_y = 420))
#   cat(i, RMSE(c, b), fill = T)
# }
# Optymalny prog - 0.39

# d <- sapply(d, function(x) threshold(x, 0.39))
# id <- read_csv('sampleSubmission.csv', col_types = 'c-')
# which(d[[1]]==min(d[[1]]), arr.ind = T)
# e <- sapply(d, function(x) as.vector(t(as.matrix(x))))
# e <- unlist(e)
# wynik <- cbind(id, value=as.numeric(e))
# write_csv(wynik, 'wynik.csv')

maks_wym <- 420

f <- lapply(a, function(x) resize(x, 540, maks_wym))
x <- array(0, dim = c(length(f), 540, maks_wym, 1))
for(i in 1:dim(x)[1]){
      x[i,,,1] <- as.matrix(f[[i]])
}

g <- lapply(a1, function(x) resize(x, size_y = maks_wym))
y <- array(0, dim = c(length(f), 540, maks_wym, 1))
for(i in 1:dim(y)[1]){
  y[i,,,1] <- as.matrix(g[[i]])
}

remove(a, a1, f, g)

model <- keras_model_sequential() %>%
  layer_conv_2d(64, 3, input_shape = c(dim(x)[2:4]), activation='relu', padding = 'same') %>%
  layer_max_pooling_2d(2) %>% 
  layer_conv_2d(128, 3, activation='relu', padding = 'same') %>% 
  layer_upsampling_2d(2) %>% 
  layer_conv_2d(1, 3, activation='sigmoid', padding = 'same')
model %>% compile(optimizer_adam(lr=0.002), 'mse')
model %>% fit(x, y, epochs = 2, batch_size = 2)
# model %>% save_model_hdf5('szumy.hdf5')
# model <- load_model_hdf5('szumy.hdf5')
d <- load.dir('test')
wymiary <- sapply(d, function(x) dim(x)[2])
d <- lapply(d, function(x) resize(x, size_y = maks_wym))
x_wal <- array(0, dim = c(length(d), 540, maks_wym, 1))
for(i in 1:dim(x_wal)[1]){
  x_wal[i,,,1] <- as.matrix(d[[i]])
}
pred <- model %>% predict(x_wal)
obrazek <- as.cimg(pred[1,,,])
plot(obrazek)

obrazki <- as.list(rep(0, dim(pred)[1]))
for(i in 1:dim(pred)[1]){
  obrazki[[i]] <- as.cimg(pred[i,,,1])
}
