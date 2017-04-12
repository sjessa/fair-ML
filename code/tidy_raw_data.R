library(tidyverse)

# Tidy communities & cirme dataset ------------------------------------------- #
crime_header <- read.csv("../data/crime/communities_metadata.csv",
                         colClasses = c("character", "character"))
crime_raw    <- read.csv("../data/crime/communities.data",
                         col.names = crime_header$variable)

crime_clean  <- crime_raw %>% select_if(is.numeric) %>% 
  # Remove non-predictive variables
  select(-state, -fold) %>% 
  # Binarize a sensitive attribute and a class
  mutate(black = ifelse(racepctblack >= 0.06, 1, 0),
         crime = ifelse(ViolentCrimesPerPop > 0.375, 1, 0)) %>% 
  select(-racepctblack, -ViolentCrimesPerPop)

crime_clean %>% write_tsv("../data/crime/crime_clean.tsv")

# Tidy census dataset -------------------------------------------------------- #
cens_header <- read.csv("../data/census/census_header.txt", header = FALSE)
cens_header <- c(as.character(cens_header[[1]]), "income")
cens_raw1   <- read.csv("../data/census/adult.data", col.names = cens_header)
cens_raw2   <- read.csv("../data/census/adult.test", skip = 1, col.names = cens_header)

# Merge
cens_raw    <- bind_rows(cens_raw1, cens_raw2)

cens_clean  <- cens_raw %>%
  select(-fnlwgt) %>% 
  mutate(income = ifelse(income == " >50K", 1, 0))
  
cens_clean %>% write_tsv("../data/census/census_clean.tsv")
