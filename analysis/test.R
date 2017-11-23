library(tidyverse)

auc_list <- list.files("../code/output", pattern = "*auc*", full.names = TRUE) %>% 
  lapply(read_csv)

auc <- Map(function(df, nm) mutate(df, method = nm), auc_list, auc_methods) %>% 
  bind_rows

methods <- c("None", "2 Naive Bayes", "Massage", "Reweighted", "Unisample")
recid_auc <- list.files("../code/output", pattern = "*recidivism.*auc*", full.names = TRUE)
crime_auc <- setdiff(list.files("../code/output", pattern = "*auc*", full.names = TRUE), recid_auc) %>%
  tail(5) %>% 
  lapply(read_csv)
  
Map(function(df, nm) mutate(df, method = nm), crime_auc, methods) %>%
  lapply(select, Model = classifier, Method = method, AUC = auc) %>% 
  lapply(mutate, Method = factor(Method)) %>%
  bind_rows() %>% 
  spread(Model, AUC)


  lapply(mutate, Method = case_when(
    .$Method == "baseline" ~ "None",
    .$Method == "massage" ~ "Massaging",
    .$Method == "reweighed" ~ "Reweighting",
    .$Method == "unisample" ~ "Uniform sampling",
    .$Method == "2nb" ~ "2 Naive Bayes"
  ))
