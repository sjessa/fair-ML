#
# Implementation of the three discrimination measures in R for ease in analysis
#
# s1 is the protected group for the sensitive characteristic, s0 the complement
# y1 is the favoured outcome, y0 the complement
#

count <- function(df, protected_variable, protected_group, outcome, fav_outcome) {
  # Helper function to tabulate counts in combos of protected_variable and prediction
  
  counts <- df %>%
    group_by_(protected_variable, outcome) %>%
    summarise(n = n()) %>% 
    as.data.frame()
  
  return(counts)
}


disc <- function(df, measure, protected_variable, protected_group, outcome, fav_outcome) {
  
  counts <- count(df, protected_variable, protected_group, outcome, fav_outcome)
  
  p_ypos_s1 <- counts[(counts[protected_variable] == protected_group) & (counts[outcome] == fav_outcome), "n"] /
    sum(counts[counts[protected_variable] == protected_group, "n"])
  
  p_ypos_s0 <- counts[(counts[protected_variable] != protected_group) & (counts[outcome] == fav_outcome), "n"] /
    sum(counts[counts[protected_variable] != protected_group, "n"])
  
  p_ypos <- sum(counts[counts[outcome] == fav_outcome, "n"])/nrow(df)
  
  p_yneg_s1 <- counts[(counts[protected_variable] == protected_group) & (counts[outcome] != fav_outcome), "n"] /
    sum(counts[counts[protected_variable] == protected_group, "n"])
  
  p_yneg_s0 <- counts[(counts[protected_variable] != protected_group) & (counts[outcome] != fav_outcome), "n"] /
    sum(counts[counts[protected_variable] != protected_group, "n"])
  
  if(measure == "elift") { # r = p(y+|s1)/p(y+)
    
    r <- p_ypos_s1/p_ypos
    if(length(r) == 0) r <- NA
    else if(any(is.na(r))) r <- NA
    
  } else if(measure == "impact") {  # r = p(y+|s1)/p(y+|s0)
    
    r <- p_ypos_s1/p_ypos_s0
    if(length(r) == 0) r <- NA
    else if(any(is.na(r))) r <- NA
  
  } else if(measure == "odds") { # r = p(y+|s0)p(y−|s1) / p(y+|s1)p(y−|s0)
    
    r <- (p_ypos_s1 * p_yneg_s0) / (p_ypos_s0 * p_yneg_s1)
    if(length((p_ypos_s0 * p_yneg_s1)) == 0) r <- NA
    else if(any(is.na(r))) r <- NA
    
  }
  
  return(r)
  
}


getaccuracy <- function(df, prediction, target) {
  
  nrow(df[df[target] == df[prediction], ]) / nrow(df)
  
}