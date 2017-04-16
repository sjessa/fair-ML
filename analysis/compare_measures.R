# 
# Barplot of accuracy scores and discrim measures for discrimination-reduction methods
#

library(tidyverse)

acc <- read_csv("../code/metrics/accuracy.csv") %>% 
  separate(X1, into = c("method", "drop", "model"), sep = "_") %>% 
  separate(method, into = c("drop1", "drop2", "method"), sep = "/") %>% 
  separate(model, into = c("model", "drop3")) %>% 
  select(-matches("drop"), -X6,
         accuracy = 7, precision = 8, recall = 9, F1 = 10) %>% 
  mutate(method = case_when(
    .$method == "baseline" ~ "None",
    .$method == "massage" ~ "Massaging",
    .$method == "reweighed" ~ "Reweighting",
    .$method == "unisample" ~ "Uniform sampling",
    .$method == "2nb" ~ "2 Naive Bayes"
  ))

acc_plot <- acc %>% gather(statistic, value, 3:6) %>%
  ggplot(aes(x = statistic, y = value)) +
  geom_bar(stat = "identity", aes(fill = method), position = "dodge") +
  xlab(NULL) +
  ggtitle("Accuracy scores for discrimination-reduction methods") +
  guides(fill = guide_legend(title = "Method")) +
  facet_wrap(~ model) +
  theme_bw()

ggsave("../figures/accuracy.pdf", width = 10, height = 3.5)

comp <- read_csv("../code/metrics/comparison.csv") %>% 
  mutate(type = c(rep("dataset", 3), rep("predictions", 10))) %>% 
  mutate(method = case_when(
    grepl("baseline", .$X1) ~ "None",
    grepl("massage", .$X1) ~ "Massaging",
    grepl("reweighed", .$X1) ~ "Reweighting",
    grepl("unisample", .$X1) ~ "Uniform sampling",
    grepl("2nb", .$X1) ~ "2 Naive Bayes",
    TRUE ~ "None"
  )) %>% 
  mutate(model = c("dataset", "dataset", "dataset", "logr", "svm", "logr", "logr", "gnb", "gnb", "gnb", "svm", "svm", "gnb")) %>% 
  select(type, model, method, impact_ratio = 2, elift_ratio = 3, odds_ratio = 4)

comp$method = factor(comp$method, levels = c("None", "Massaging", "Reweighting", "Uniform sampling", "2 Naive Bayes"))

comp_plot <- comp %>% gather(measure, value, 4:6) %>%
  ggplot(aes(x = method, y = value)) +
  geom_bar(stat = "identity", aes(fill = model), position = "dodge") +
  guides(fill = guide_legend(title = "Model")) +
  xlab("Method") +
  facet_wrap(~ measure) +
  ggtitle("Comparison of discrimination measures on raw data and after interventions") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

ggsave("../figures/measures.pdf", width = 10, height = 3.5)

