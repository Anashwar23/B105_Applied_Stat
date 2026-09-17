## =====================================================================
## The Relationship Between Advertising Spending and Sales
## Business problem: how should a firm allocate an advertising budget
##                    across TV, radio and newspaper to maximise sales?
## Data source: "Advertising" dataset (James et al., 2013), 200 markets,
## mirrored on GitHub at:
## https://raw.githubusercontent.com/selva86/datasets/master/
##   Advertising.csv
## Original source: https://www.statlearning.com/s/Advertising.csv
## Documented at: https://search.r-project.org/CRAN/refmans/glmtoolbox/
##   html/advertising.html
## =====================================================================

set.seed(2026)
suppressMessages({
  library(ggplot2)
  library(dplyr)
  library(car)      # vif(), durbinWatsonTest()
  library(lmtest)   # bptest() - Breusch-Pagan test
})

setwd("~/advertising_project")

## ---------------------------------------------------------------
## 1. LOAD & INITIAL INSPECTION
## ---------------------------------------------------------------
raw <- read.csv("data/Advertising.csv", stringsAsFactors = FALSE)
cat("Raw dimensions:", dim(raw), "\n")
str(raw)
colSums(is.na(raw))
sum(duplicated(raw))

## ---------------------------------------------------------------
## 2. DATA CLEANING & PREPROCESSING
## ---------------------------------------------------------------
df <- raw
df$X <- NULL                                  # drop redundant row-index column
names(df) <- c("TV", "radio", "newspaper", "sales")   # consistent, tidy names

# Range / plausibility checks (spend and sales cannot be negative)
stopifnot(all(df >= 0))
summary(df)
cat("Final dimensions after cleaning:", dim(df), "\n")

write.csv(df, "outputs/advertising_clean.csv", row.names = FALSE)

## ---------------------------------------------------------------
## 3. SAMPLING: TRAIN / TEST HOLD-OUT SPLIT
## ---------------------------------------------------------------
# With only 200 observations, no subsampling is needed for the
# inferential tests (the full population is used for hypothesis
# testing in Section 6). To additionally demonstrate sampling
# technique and to check that the fitted model generalises rather
# than overfitting the 200 markets on hand, an 80/20 simple random
# split (160 training / 40 test markets) is drawn, and prediction
# accuracy on the held-out 40 markets is reported in Section 7.5.
n <- nrow(df)
train_idx <- sample(seq_len(n), size = round(0.8 * n))
train <- df[train_idx, ]
test  <- df[-train_idx, ]
cat("Training set:", nrow(train), " | Test set:", nrow(test), "\n")

## ---------------------------------------------------------------
## 4. EXPLORATORY DATA ANALYSIS
## ---------------------------------------------------------------
desc <- df %>% summarise(
  across(everything(), list(mean = mean, sd = sd, median = median,
                             min = min, max = max))
)
print(t(desc))
write.csv(t(desc), "outputs/descriptive_stats.csv")

cor_matrix <- cor(df)
print(round(cor_matrix, 3))
write.csv(round(cor_matrix, 3), "outputs/correlation_matrix.csv")

p1 <- ggplot(df, aes(TV, sales)) + geom_point(colour = "steelblue") +
  geom_smooth(method = "lm", se = FALSE, colour = "darkred") + theme_minimal() +
  labs(title = "Sales vs TV Advertising Spend", x = "TV spend ($000s)", y = "Sales (000s units)")
ggsave("outputs/fig1_sales_tv.png", p1, width = 5.5, height = 4, dpi = 150)

p2 <- ggplot(df, aes(radio, sales)) + geom_point(colour = "seagreen") +
  geom_smooth(method = "lm", se = FALSE, colour = "darkred") + theme_minimal() +
  labs(title = "Sales vs Radio Advertising Spend", x = "Radio spend ($000s)", y = "Sales (000s units)")
ggsave("outputs/fig2_sales_radio.png", p2, width = 5.5, height = 4, dpi = 150)

p3 <- ggplot(df, aes(newspaper, sales)) + geom_point(colour = "orange3") +
  geom_smooth(method = "lm", se = FALSE, colour = "darkred") + theme_minimal() +
  labs(title = "Sales vs Newspaper Advertising Spend", x = "Newspaper spend ($000s)", y = "Sales (000s units)")
ggsave("outputs/fig3_sales_newspaper.png", p3, width = 5.5, height = 4, dpi = 150)

p4 <- ggplot(df, aes(sales)) + geom_histogram(bins = 15, fill = "grey40") +
  theme_minimal() + labs(title = "Distribution of Sales", x = "Sales (000s units)")
ggsave("outputs/fig4_sales_hist.png", p4, width = 5.5, height = 4, dpi = 150)

## ---------------------------------------------------------------
## 5. INFERENTIAL TESTS: SIMPLE CORRELATIONS (H1-H3)
## ---------------------------------------------------------------
cor_tv    <- cor.test(df$TV, df$sales)
cor_radio <- cor.test(df$radio, df$sales)
cor_news  <- cor.test(df$newspaper, df$sales)
print(cor_tv); print(cor_radio); print(cor_news)

## ---------------------------------------------------------------
## 6. MULTIPLE LINEAR REGRESSION (H4) + ASSUMPTION CHECKS
## ---------------------------------------------------------------
model <- lm(sales ~ TV + radio + newspaper, data = train)
summary(model)
confint(model)

# 6.1 Linearity: fitted-vs-residual plot should show no pattern
png("outputs/fig5_residuals_fitted.png", width = 700, height = 500, res = 150)
plot(model, which = 1)
dev.off()

# 6.2 Normality of residuals
shapiro.test(resid(model))
png("outputs/fig6_qq_residuals.png", width = 700, height = 500, res = 150)
plot(model, which = 2)
dev.off()

# 6.3 Homoscedasticity: Breusch-Pagan test
bptest(model)

# 6.4 Independence of errors: Durbin-Watson test
durbinWatsonTest(model)

# 6.5 Multicollinearity: Variance Inflation Factor
vif(model)

## ---------------------------------------------------------------
## 7. INTERPRETING THE MODEL
## ---------------------------------------------------------------
# 7.1 Overall model significance already in summary() (F-statistic)
# 7.2 Standardised coefficients, to compare relative importance
train_std <- as.data.frame(scale(train))
model_std <- lm(sales ~ TV + radio + newspaper, data = train_std)
cat("Standardised coefficients:\n"); print(coef(model_std))

# 7.3 Simple-vs-multiple regression comparison for newspaper
model_news_only <- lm(sales ~ newspaper, data = train)
summary(model_news_only)

# 7.4 Model excluding the non-significant newspaper predictor
model_reduced <- lm(sales ~ TV + radio, data = train)
summary(model_reduced)
anova(model_reduced, model)   # nested-model F-test: does newspaper add explanatory power?

# 7.5 Out-of-sample validation on the held-out 20% test set
pred_full    <- predict(model, newdata = test)
pred_reduced <- predict(model_reduced, newdata = test)
rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
r2_oos <- function(actual, pred) 1 - sum((actual-pred)^2) / sum((actual-mean(actual))^2)
cat("Full model   - Test RMSE:", round(rmse(test$sales, pred_full),3),
    " | Test R2:", round(r2_oos(test$sales, pred_full),3), "\n")
cat("Reduced model- Test RMSE:", round(rmse(test$sales, pred_reduced),3),
    " | Test R2:", round(r2_oos(test$sales, pred_reduced),3), "\n")

## ---------------------------------------------------------------
## 8. SAVE KEY NUMERIC RESULTS FOR THE REPORT
## ---------------------------------------------------------------
sink("outputs/model_summary.txt")
cat("=== Pearson correlations with sales ===\n")
print(cor_tv); print(cor_radio); print(cor_news)
cat("\n=== Full multiple regression (training set) ===\n"); print(summary(model))
cat("\n=== 95% CIs for coefficients ===\n"); print(confint(model))
cat("\n=== Assumption checks ===\n")
cat("Shapiro-Wilk (residuals):\n"); print(shapiro.test(resid(model)))
cat("Breusch-Pagan:\n"); print(bptest(model))
cat("Durbin-Watson:\n"); print(durbinWatsonTest(model))
cat("VIF:\n"); print(vif(model))
cat("\n=== Reduced model (TV + radio only) ===\n"); print(summary(model_reduced))
cat("\n=== Nested model F-test (newspaper's added value) ===\n")
print(anova(model_reduced, model))
cat("\n=== Out-of-sample test-set performance ===\n")
cat("Full model    RMSE:", rmse(test$sales, pred_full),
    " R2:", r2_oos(test$sales, pred_full), "\n")
cat("Reduced model RMSE:", rmse(test$sales, pred_reduced),
    " R2:", r2_oos(test$sales, pred_reduced), "\n")
sink()

cat("\nAnalysis complete. Outputs written to /outputs.\n")
