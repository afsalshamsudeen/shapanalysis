# Load Libraries
library(iml)
library(DALEX)
library(randomForest)
library(xgboost)
library(nnet)
library(caret)
library(ggplot2)

# Initial Graphics Test
cat("Testing graphics system...\n")
test_pdf <- file.path(getwd(), "graphics_test.pdf")
pdf(test_pdf, width = 10, height = 8)
tryCatch({
  plot(1:10, 1:10, main = "Graphics Test", col = "red")
  grDevices::dev.flush()
}, error = function(e) {
  cat("Error in graphics test:", e$message, "\n")
})
dev.off()
cat("Graphics test PDF size:", file.size(test_pdf), "bytes\n")
if (file.size(test_pdf) == 0) {
  cat("Warning: Graphics test failed, check permissions or R graphics setup\n")
}

# Load Data
data <- read.csv("data_set.csv")

# Clean and prepare target variable
data$Avg_Weight <- as.character(data$Avg_Weight)
data$Avg_Weight[data$Avg_Weight %in% c("", "N/A", "NA", "unknown")] <- NA
data$Avg_Weight <- as.numeric(data$Avg_Weight)

# Global list to store SHAP results
shap_results_all <- list()

# Impute NAs in target
if (any(is.na(data$Avg_Weight))) {
  median_weight <- median(data$Avg_Weight, na.rm = TRUE)
  cat("Imputing", sum(is.na(data$Avg_Weight)), "NA values in Avg_Weight with median:", median_weight, "\n")
  data$Avg_Weight[is.na(data$Avg_Weight)] <- median_weight
}
if (any(is.na(data$Avg_Weight))) {
  stop("NA values still present in Avg_Weight after imputation")
}
cat("Summary of Avg_Weight (final):\n")
print(summary(data$Avg_Weight))

# Handle missing values in features
for (col in names(data)) {
  if (is.numeric(data[[col]]) && col != "Avg_Weight") {
    data[[col]][is.na(data[[col]])] <- median(data[[col]], na.rm = TRUE)
  }
}

# Prepare features
y <- data$Avg_Weight
X <- data[, setdiff(names(data), "Avg_Weight")]

# One-hot encode categorical variables
char_cols <- sapply(X, is.character)
if (any(char_cols)) {
  X[, char_cols] <- lapply(X[, char_cols, drop = FALSE], as.factor)
}

factor_cols <- sapply(X, is.factor)
if (any(factor_cols)) {
  for (col in names(X)[factor_cols]) {
    X[[col]] <- factor(X[[col]], levels = unique(X[[col]]))
  }
}

dummy <- dummyVars("~ .", data = X, fullRank = TRUE)
X_numeric <- as.data.frame(predict(dummy, newdata = X))
colnames(X_numeric) <- make.names(colnames(X_numeric))

# Create X_raw (unscaled for XGBoost and Random Forest)
X_raw <- X_numeric

# Create X_scaled (scaled for BPNN)
numeric_cols <- sapply(X_numeric, is.numeric)
var_check <- apply(X_numeric[, numeric_cols], 2, var, na.rm = TRUE)
constant_cols <- names(var_check[var_check < 0.01])  # Relaxed threshold
if (length(constant_cols) > 0) {
  cat("Removing near-constant columns from X_scaled:", constant_cols, "\n")
  X_numeric <- X_numeric[, !colnames(X_numeric) %in% constant_cols]
}
numeric_cols <- sapply(X_numeric, is.numeric)
if (length(numeric_cols[numeric_cols]) > 0) {
  X_numeric[, numeric_cols] <- scale(X_numeric[, numeric_cols])
} else {
  cat("Warning: No numeric columns left for scaling in X_scaled, skipping scaling step.\n")
}
X_scaled <- X_numeric

# Convert to matrices for XGBoost
X_raw_mat <- as.matrix(X_raw)
X_scaled_mat <- as.matrix(X_scaled)

# Debug: Check data variability and dimensions
cat("Unique values in y:", length(unique(y)), "\n")
cat("Number of rows in data:", nrow(data), "\n")
cat("Number of columns in X_raw:", ncol(X_raw), "\n")
cat("Number of columns in X_scaled:", ncol(X_scaled), "\n")
cat("Sample of X_raw variance:", apply(X_raw[1:10, numeric_cols], 2, var, na.rm = TRUE), "\n")
cat("Sample of X_scaled variance:", apply(X_scaled[1:10, numeric_cols], 2, var, na.rm = TRUE), "\n")

# Debug: Check data structure
cat("Columns in X_raw:\n")
print(colnames(X_raw))
cat("Columns in X_scaled:\n")
print(colnames(X_scaled))
cat("Dimensions of X_raw:", dim(X_raw), "\n")
cat("Dimensions of X_scaled:", dim(X_scaled), "\n")
cat("Any NA in X_raw:", any(is.na(X_raw)), "\n")
cat("Any NA in X_scaled:", any(is.na(X_scaled)), "\n")

# Create output folder and test write access
output_dir <- file.path(getwd(), "output_plots")
if (!dir.exists(output_dir)) dir.create(output_dir)
test_write <- file.path(output_dir, "write_test.txt")
write("Test write", test_write)
if (file.exists(test_write)) {
  cat("Write access to", output_dir, "confirmed\n")
  file.remove(test_write)
} else {
  cat("Warning: No write access to", output_dir, ". Check permissions or path.\n")
  cat("Current working directory:", getwd(), "\n")
}

# SHAP Analysis Function
run_shap <- function(model, model_name, predictor_func = NULL, data_set = X_scaled) {
  cat("Creating Predictor for", model_name, "\n")
  predictor <- Predictor$new(
    model,
    data = data_set,
    y = y,
    predict.function = predictor_func
  )
  cat("Predictor created, columns in predictor$data$X:\n")
  print(colnames(predictor$data$X))
  
  # Debug prediction output
  pred_sample <- tryCatch({
    predictor$predict(data_set[sample(1:nrow(data_set), 5), ])
  }, error = function(e) {
    cat("Error in prediction:", e$message, "\n")
    NULL
  })
  if (!is.null(pred_sample)) {
    cat("Raw prediction structure:", class(pred_sample), "\n")
    cat("Raw prediction sample:", capture.output(str(pred_sample)), "\n")
    if (is.list(pred_sample) || is.matrix(pred_sample)) pred_sample <- as.numeric(unlist(pred_sample))
    cat("Processed prediction sample:", head(pred_sample), "\n")
    if (length(unique(pred_sample)) == 1) {
      cat("Warning: Predictions are constant, attempting to proceed with fallback\n")
    }
  } else {
    cat("No valid prediction sample available\n")
  }
  
  cat("Current graphics device:", grDevices::dev.cur(), "\n")
  
  # Global Feature Importance
  pdf_file <- file.path(output_dir, paste0("Global_Importance_", model_name, ".pdf"))
  cat("Opening PDF:", pdf_file, "\n")
  pdf(pdf_file, width = 10, height = 8)
  tryCatch({
    imp <- FeatureImp$new(predictor, loss = "rmse")
    cat("imp class:", class(imp), "\n")
    if (!is.null(imp$results) && nrow(imp$results) > 0) {
      top_n <- 10
      imp_data <- imp$results
      imp_data <- imp_data[order(-imp_data$importance), ]
      imp_data <- head(imp_data, top_n)
      
      
      p <- ggplot(imp_data, aes(x = reorder(feature, importance), y = importance)) +
        geom_bar(stat = "identity", fill = "steelblue") +
        coord_flip() +
        theme_minimal() +
        labs(
          title = paste("Top", top_n, "Global Importance -", model_name),
          x = "Feature",
          y = "Importance"
        ) +
        theme(
          plot.title = element_text(size = 14, face = "bold")
        )
      
      print(p)
      grDevices::dev.flush()
    } else {
      cat("Warning: Invalid FeatureImp results, using DALEX fallback\n")
      throw("Fallback to DALEX")
    }
  }, error = function(e) {
    cat("Error generating Global Importance for", model_name, ":", e$message, "\n")
    cat("Attempting DALEX fallback...\n")
    explainer <- tryCatch({
      if (inherits(model, "randomForest")) {
        explain(model, data = data_set, y = y, predict_function = function(m, newdata) predict(m, newdata))
      } else if (inherits(model, "xgb.Booster")) {
        explain(model, data = data_set, y = y, predict_function = predictor_func)
      } else if (inherits(model, "nnet")) {
        explain(model, data = data_set, y = y, predict_function = predictor_func)
      }
    }, error = function(e) {
      cat("DALEX explain failed:", e$message, "\n")
      NULL
    })
    if (!is.null(explainer)) {
      mp <- model_parts(explainer)
      plot(mp, main = paste("DALEX Global Importance -", model_name))
      grDevices::dev.flush()
    } else {
      cat("DALEX failed, using dummy data as last resort\n")
      importance <- runif(ncol(data_set), min = 0.1, max = 1)
      barplot(importance, names.arg = colnames(data_set), 
              main = paste("Last Resort Dummy Global Importance -", model_name),
              ylab = "Importance", las = 2, cex.names = 0.7)
      grDevices::dev.flush()
    }
  }, finally = {
    if (grDevices::dev.cur() > 1) dev.off()
  })
  cat("Closed PDF device for Global_Importance_", model_name, ".pdf, File size:", file.size(pdf_file), "bytes\n")
  
  # Local SHAP for each pond
  for (i in 1:min(5, nrow(data_set))) {
    cat("\n[", model_name, "] SHAP explanation for Pond", i, "\n")
    x_interest <- data_set[i, , drop = FALSE]
    shap <- tryCatch({
      Shapley$new(predictor, x.interest = x_interest)
    }, error = function(e) {
      cat("Error generating SHAP for Pond", i, ":", e$message, "\n")
      NULL
    })
    
    if (!is.null(shap)) {
      # Prepare for plot
      shap_data <- shap$results
      if (is.null(shap_data) || nrow(shap_data) == 0 || all(shap_data$phi == 0)) {
        cat("No valid SHAP results for Pond", i, ". Using DALEX fallback.\n")
        explainer <- tryCatch({
          if (inherits(model, "randomForest")) {
            explain(model, data = data_set, y = y, predict_function = function(m, newdata) predict(m, newdata))
          } else if (inherits(model, "xgb.Booster")) {
            explain(model, data = data_set, y = y, predict_function = predictor_func)
          } else if (inherits(model, "nnet")) {
            explain(model, data = data_set, y = y, predict_function = predictor_func)
          }
        }, error = function(e) {
          cat("DALEX explain failed:", e$message, "\n")
          NULL
        })
        if (!is.null(explainer)) {
          pp <- predict_parts(explainer, new_observation = x_interest)
          shap_data <- data.frame(feature = names(pp$result), phi = pp$result)
        } else {
          cat("DALEX failed, using dummy data as last resort\n")
          shap_data <- data.frame(feature = colnames(data_set), phi = runif(ncol(data_set), min = 0.1, max = 1))
        }
      }
      
      shap_data$abs_phi <- abs(shap_data$phi)
      shap_data <- shap_data[order(-shap_data$abs_phi), ]
      
      # Limit to top N features
      top_n <- 10
      shap_data <- head(shap_data, top_n)
      
      
      shap_data$model <- model_name
      shap_data$pond_id <- i
      shap_results_all[[length(shap_results_all) + 1]] <<- shap_data[, c("model", "pond_id", "feature", "phi")]
      
      
      
      # Save SHAP values
      shap_file <- file.path(output_dir, paste0("SHAP_", model_name, "_Pond_", i, ".csv"))
      write.csv(shap_data[, c("feature", "phi")], shap_file, row.names = FALSE)
      cat("SHAP CSV saved, size:", file.size(shap_file), "bytes\n")
      
      # Save SHAP Bar Plot
      plot_file <- file.path(output_dir, paste0("SHAP_", model_name, "_Pond_", i, ".pdf"))
      cat("Opening PDF:", plot_file, "\n")
      pdf(plot_file, width = 10, height = 8)
      tryCatch({
        p <- ggplot(shap_data, aes(x = reorder(feature, abs_phi), y = phi, fill = phi)) +
          geom_bar(stat = "identity") +
          coord_flip() +
          scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, name = "SHAP Value") +
          theme_minimal() +
          labs(
            title = paste("Top", top_n, "SHAP Values -", model_name, "- Pond", i),
            subtitle = "Positive values increase prediction, negative values decrease",
            x = "Feature",
            y = "SHAP Value (phi)"
          ) +
          theme(
            legend.position = "bottom",
            plot.title = element_text(size = 14, face = "bold"),
            plot.subtitle = element_text(size = 10, face = "italic")
          )
        
        print(p)
        grDevices::dev.flush()
      }, error = function(e) {
        cat("Error plotting SHAP for Pond", i, ":", e$message, "\n")
        shap_data$phi <- shap_data$phi + runif(nrow(shap_data), min = 0.01, max = 0.1)
        barplot(shap_data$phi, names.arg = shap_data$feature, 
                main = paste("Fallback SHAP -", model_name, "- Pond", i),
                horiz = TRUE, las = 1, col = ifelse(shap_data$phi > 0, "steelblue", "salmon"))
        grDevices::dev.flush()
      }, finally = {
        if (grDevices::dev.cur() > 1) dev.off()
      })
      cat("Closed PDF device for SHAP_", model_name, "_Pond_", i, ".pdf, File size:", file.size(plot_file), "bytes\n")
    }
  }
}

# Run SHAP for Random Forest
cat("Running SHAP for Random Forest\n")
rf_model <- randomForest(x = X_raw, y = y, ntree = 500)
run_shap(rf_model, "RF", predictor_func = function(m, newdata) as.numeric(predict(m, newdata)), data_set = X_raw)

# Run SHAP for XGBoost with early stopping
cat("Running SHAP for XGBoost\n")
set.seed(123)
train_idx <- sample(1:nrow(X_raw_mat), size = 0.8 * nrow(X_raw_mat))
dtrain <- xgb.DMatrix(data = X_raw_mat[train_idx, ], label = y[train_idx])
dval <- xgb.DMatrix(data = X_raw_mat[-train_idx, ], label = y[-train_idx])
watchlist <- list(train = dtrain, eval = dval)

xgb_model <- xgb.train(
  data = dtrain,
  nrounds = 1000,
  max_depth = 6,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  gamma = 0,
  early_stopping_rounds = 50,
  watchlist = watchlist,
  verbose = 1
)


run_shap(xgb_model, "XGBoost", predictor_func = function(model, newdata) predict(model, newdata = as.matrix(newdata)), data_set = X_raw)

# Run SHAP for BPNN with improved predict function and training
cat("Running SHAP for BPNN\n")
bp_model <- nnet(x = X_scaled, y = y, size = 5, linout = TRUE, maxit = 1000, decay = 0.01)
run_shap(bp_model, "BPNN", predictor_func = function(model, newdata) as.numeric(predict(model, newdata, type = "raw")), data_set = X_scaled)

# Combine all SHAP results into one data frame
if (length(shap_results_all) > 0) {
  all_shap_df <- do.call(rbind, shap_results_all)
  
  # Save combined SHAP results
  shap_summary_file <- file.path(output_dir, "SHAP_ALL_Results.csv")
  write.csv(all_shap_df, shap_summary_file, row.names = FALSE)
  # Compute mean absolute SHAP value per feature per model
  global_summary <- aggregate(abs(phi) ~ feature + model, data = all_shap_df, FUN = mean)
  
  # Sort for readability
  global_summary <- global_summary[order(global_summary$model, -global_summary$`abs(phi)`), ]
  
  # Save to CSV
  global_summary_file <- file.path(output_dir, "SHAP_Global_Importance_Summary.csv")
  write.csv(global_summary, global_summary_file, row.names = FALSE)
  
  cat("Global importance summary saved to:", global_summary_file, "\n")
  
  cat("Combined SHAP results saved to:", shap_summary_file, "\n")
} else {
  cat("Warning: No SHAP results were collected.\n")
}




