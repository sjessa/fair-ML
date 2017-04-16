library(tidyverse)

recid <- read_csv("../data/recidivism/recidivism.csv") %>% select(-1)
names(recid) <- c("year", "sex", "offense_class", "offense_type", "offense_subtype",
                  "release_type", "recidivism_type", "days_to_recidivism",
                  "offense2_class", "offense2_type", "offense2_subtype", "target_pop",
                  "histpanic", "non_white", "age", "recidivism")

recid_tidy <- recid %>% select(sex, offense_class, offense_type, offense_subtype,
                               age, non_white, recidivism)

write_csv(recid_tidy, "../data/recidivism/recidivism_clean.csv")