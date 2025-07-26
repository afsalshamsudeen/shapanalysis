# Load Libraries
library(iml)
library(DALEX)
library(randomForest)
library(xgboost)
library(nnet)
library(caret)
library(ggplot2)

# ... [preprocessing and model training code remains unchanged] ...

# SHAP Analysis Function (Improved for clearer plots)
run_shap <- function(model, model_name, predictor_func = NULL, data_set = X_scaled) {
  cat("Creating Predictor for", model_name, "\n")
  predictor <- Predictor$new(
    model,
    data = data_set,
    y = y,
    predict.function = predictor_func
  )
  
  # Global Feature Importance (top 10 features, improved plot)
  pdf_file <- file.path(output_dir, paste0("Global_Importance_", model_name, ".pdf"))
  cat("Opening PDF:", pdf_file, "\n")
  pdf(pdf_file, width = 10, height = 8)
  tryCatch({
    imp <- FeatureImp$new(predictor, loss = "rmse")
    imp_data <- imp$results
    if (!is.null(imp_data) && nrow(imp_data) > 0) {
      imp_data <- imp_data[order(-imp_data$importance), ]
      top_imp <- head(imp_data, 10)
      p <- ggplot(top_imp, aes(x = reorder(feature, importance), y = importance)) +
        geom_bar(stat = "identity", fill = "steelblue") +
        coord_flip() +
        theme_minimal() +
        labs(title = paste("Top 10 Global Feature Importance -", model_name),
             subtitle = "Based on RMSE loss",
             x = "Feature",
             y = "Importance") +
        theme(legend.position = "none")
      print(p)
    }
  }, finally = {
    dev.off()
  })
  
  # Local SHAP Explanations (top 10 features, improved plots)
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
      shap_data <- shap$results
      shap_data$abs_phi <- abs(shap_data$phi)
      shap_data <- shap_data[order(-shap_data$abs_phi), ]
      top_shap <- head(shap_data, 10)
      
      # Save Plot
      plot_file <- file.path(output_dir, paste0("SHAP_", model_name, "_Pond_", i, ".pdf"))
      pdf(plot_file, width = 10, height = 8)
      tryCatch({
        p <- ggplot(top_shap, aes(x = reorder(feature, abs_phi), y = phi, fill = phi)) +
          geom_bar(stat = "identity") +
          coord_flip() +
          scale_fill_gradient2(low = "salmon", mid = "white", high = "steelblue", midpoint = 0) +
          theme_minimal() +
          labs(title = paste("Top 10 SHAP Values -", model_name, "- Pond", i),
               subtitle = "Impact on prediction",
               x = "Feature",
               y = "SHAP Value (phi)") +
          theme(legend.position = "bottom")
        print(p)
      }, finally = {
        dev.off()
      })
    }
  }
}
