library(tidyverse)
library(ggplot2)
library(glmnet)
library(broom)
library(patchwork)
library(rsample)
library(recipes)
library(purrr)
library(yardstick)
library(dbarts)
library(randomForest)
library(pROC)

# loading data (no headers) need for column names

col_names <- c("age", "gender", "total_bilirubin", "direct_bilirubin", "alkphos",
               "sgpt", "sgot", "total_proteins", "albumin", "agratio", "class")


df <- read.csv("data/ildp.csv", header = FALSE, col.names = col_names)

# dataset structure
str(df)
dim(df)
glimpse(df)

# checking for missing values
colSums(is.na(df))

# ag ratio is the only predictor with 4 missing values thus 4 missing observations for the entire datasheet
# percentage of missing values
(sum(is.na(df$agratio)) / nrow(df)) * 100

# missing observations are 0.68% of to the total data set so we can do a complete case analysis

# dropping the missing values for complete case analysis
df <- df %>% 
  drop_na(agratio)

dim(df)

# recoding variables to appropriate data type
# class and gender are stored as integer data type instead of factor
df <- df %>% 
  mutate(class = factor(class, labels = c("Liver Disease", "No Disease")),
         gender = factor(gender, labels = c("Female", "Male"))
         )

## Descriptive statistics 

#  numerical columns for histograms and box plots
 num_cols <- c("age", "total_bilirubin", "direct_bilirubin",
               "total_proteins", "albumin", "agratio", "sgpt", "sgot", "alkphos")
 
sum_stats <- function(x){
   c(Mean = mean(x), SD= sd(x))
 }
 
sapply(df[, num_cols], sum_stats)

aggregate(. ~ class, data = df[, c(num_cols, "class")],
          function(x) c(mean=mean(x), sd = sd(x)))
 
# class balance check
table(df$class)

#####################################################

# Feature distributions  histoggram and boxplots

df_long <- df %>% 
  select(all_of(num_cols), class) %>% 
  pivot_longer(-class, names_to = "feature", values_to = "value")

hist_plot<- ggplot(df_long, aes(x = value, fill = class)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  facet_wrap(~feature, scales = "free", ncol = 3) +
  labs(title = "Feature Distribution by class",
       x = NULL, y = "Count", fill = "Class") +
  # scale_y_continuous(breaks = NULL) +
  # scale_x_continuous(breaks = NULL) +
  theme_minimal()

ggsave("hist_plot.png", plot = hist_plot, width = 6, height = 4, dpi = 300)


# box plot
box_plot <- ggplot(df_long, aes(x = class, y = value, fill = class)) +
  geom_boxplot(outlier.size = 0.8, alpha = 0.7 ) +
  facet_wrap(~feature, scales = "free_y", ncol = 3) +
  labs(title = "Feature Distribution by class", 
       x = NULL, y = NULL, fill = "Class")+
  theme_minimal()

box_plot
ggsave("box_plot.png", plot = box_plot, width = 6, height = 4, dpi = 300)

# correlation

cor_data <- df %>% 
  select(all_of(num_cols)) %>% 
  cor() %>% 
  as.data.frame() %>% 
  rownames_to_column("var1") %>% 
  pivot_longer(-var1, names_to = "var2", values_to = "correlation")

heat_map <- ggplot(cor_data, aes(x = var1, y = var2, fill = correlation))+
  geom_tile(color = "white") +
  geom_text(aes(label = round(correlation, 2)), size = 3) +
  scale_fill_gradient2(low = "#5C9BE0", mid = "white", high = "#E05C5C",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(title = "Correlation Heatmap", x = NULL, y = NULL) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

heat_map
ggsave("heat_map.png", plot = heat_map, width = 6, height = 4, dpi = 300)



# Modelling 
df <- df %>% 
  mutate(gender = as.integer(gender=="Male"))

# train/test split and cross validation set up
set.seed(25250170)

split <- initial_split(df, prop = 0.7, strata = class)
train_data <- training(split)
test_data <- testing(split)

# checking if the splitting was a win
dim(train_data)
dim(test_data)

# stratified 5-fold CV on the training data for cross validation
folds <- vfold_cv(train_data, v= 5, strata = class)

# the training data has 404 instances in total and a 5 fold cross validation split
# has a 323 instances meant for training and 81 observations for testing 

# Cross Validation for Random Forests
cv_rf <- function(split){
  
  tr <- training(split)
  ts <- testing(split)
  
  fit <- randomForest(class ~ ., data = tr, ntree = 500)
  probs <- predict(fit, ts, type ="prob")[, "Liver Disease"]
  
  tibble(truth = ts$class, estimate = probs)
}

cv_results_rf <- map(folds$splits, cv_rf) %>% 
  bind_rows()

cv_results_rf
uc_rf   <- roc_auc(cv_results_rf,   truth, estimate, event_level = "first")
uc_rf

# Bayesian addititive trees
# bart requires that allpredictor variables be numeric, need to gender to integer type


df <- df %>% 
  mutate(gender = as.integer(gender=="Male"))

# cross validation function for bart
cv_bart <- function(split) {
  tr <- training(split)
  v1 <- testing(split)
  
  x_tr <- tr %>%
    select(-class) %>%
    as.matrix()
  
  y_tr <- as.integer(tr$class == "Liver Disease")
  
  x_v1 <- v1 %>%
    select(-class) %>%
    as.matrix()
  
  fit   <- bart(x.train = x_tr, y.train = y_tr, ntree = 200,
                ndpost = 1000, nskip = 200, verbose = FALSE, keeptrees = TRUE)
  pred  <- predict(fit, x_v1)
  probs <- colMeans(pred)
  
  tibble(truth = v1$class, estimate = probs)
}

# creating an empty list, to add results of each fit for the bart in each fold
cv_results_list <- list()

for (i in seq_along(folds$splits)) {
  cv_results_list[[i]] <- cv_bart(folds$splits[[i]])
  gc()
}

cv_results_bart <- bind_rows(cv_results_list)

