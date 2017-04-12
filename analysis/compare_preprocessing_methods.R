library(tidyverse)


# ROC curves for pre-processed data & baseline classifiers ------------------- #

roc_paths <- list.files("../code/output/", pattern = "*roc*", full.names = TRUE)
roc_list <- roc_paths %>% lapply(read_csv)
roc <- Map(function(df, nm) mutate(df, method_model = nm), roc_list, roc_paths) %>% 
  bind_rows() %>%
  separate(method_model, into = c("method", "data", "model"), sep = "_") %>%
  mutate(method = substr(method, 17, nchar(method)),
         model = substr(model, 1, nchar(model)-4)) %>% 
  mutate(tpr = tpr/10) %>% 
  select(-data)

auc_methods <- c("baseline", "massage", "reweighed")
auc_list <- list.files("../code/output", pattern = "*auc*", full.names = TRUE) %>% 
  lapply(read_csv)

auc <- Map(function(df, nm) mutate(df, method = nm), auc_list, auc_methods) %>% 
  bind_rows

roc_plot <- roc %>% ggplot(aes(x = fpr, y = tpr, colour = method)) +
  geom_line() +
  facet_wrap(~ model) +
  xlab("FPR") + ylab("TPR") +
  ggtitle("ROC curves for baseline learners on raw and pre-processed data") +
  guides(colour = guide_legend(title = "Method"))
  
ggsave("../figures/roc_baseline_preprocessing.pdf", width = 10, height = 3.5)
