# Interactive version of the model comparison. Run from the repo root:
#   Rscript -e 'shiny::runApp()'
# Needs shiny, ggplot2 and jsonlite:
#   install.packages(c("shiny", "ggplot2", "jsonlite"))

library(shiny)
library(ggplot2)
library(jsonlite)

b <- fromJSON("reports/results/benchmark.json", simplifyVector = FALSE)

# The common test set holds every model and clinical score on the same
# variants (chromosome-split test set: chr8 and chr21).
labels <- list(revel = "REVEL", alphamissense = "AlphaMissense",
               cadd_phred = "CADD", cnn_onehot = "1D CNN",
               nt_lora = "NT + LoRA", logreg_kmer = "logistic regression")
keys <- c("revel", "alphamissense", "cadd_phred", "cnn_onehot",
          "nt_lora", "logreg_kmer")

scores <- do.call(rbind, lapply(keys, function(k) {
    m <- b$common[[k]]
    data.frame(
        model = labels[[k]],
        auroc = m$auroc,
        auroc_lo = m$auroc_ci[[1]], auroc_hi = m$auroc_ci[[2]],
        auprc = m$auprc,
        auprc_lo = m$auprc_ci[[1]], auprc_hi = m$auprc_ci[[2]],
        within_gene = m$within_gene_auroc
    )
}))
scores$model <- factor(scores$model, levels = rev(scores$model))

# Dataset split, from the README.
splits <- data.frame(
    split = c("Train", "Validation", "Test"),
    chromosomes = c("everything else", "chr10", "chr8, chr21"),
    benign = c(134144, 4894, 6976),
    pathogenic = c(57619, 1684, 1701)
)

ui <- fluidPage(
    titlePanel("DeepVariantEffect"),
    p("Predicting whether a missense variant is pathogenic from the DNA sequence",
      "around it. Clinical scores (REVEL, AlphaMissense, CADD) are shown next to",
      "the sequence-only models trained here."),
    p(sprintf("Scored on %s held-out test variants, common to all models.",
              format(b$n_common, big.mark = ","))),
    h4("AUROC and AUPRC (95% CI)"),
    plotOutput("auroc", height = "420px"),
    plotOutput("auprc", height = "420px"),
    h4("Full table"),
    tableOutput("score_table"),
    h4("ROC, precision-recall and calibration"),
    imageOutput("roc"),
    imageOutput("calibration"),
    h4("Dataset split"),
    tableOutput("splits")
)

server <- function(input, output) {
    output$auroc <- renderPlot({
        ggplot(scores, aes(x = model, y = auroc)) +
            geom_col(fill = "#4C72B0") +
            geom_errorbar(aes(ymin = auroc_lo, ymax = auroc_hi),
                          width = 0.3, colour = "grey40") +
            coord_flip() +
            coord_cartesian(ylim = c(0.3, 1.0)) +
            labs(x = NULL, y = "AUROC") +
            theme_minimal(base_size = 13)
    })

    output$auprc <- renderPlot({
        ggplot(scores, aes(x = model, y = auprc)) +
            geom_col(fill = "#d6604d") +
            geom_errorbar(aes(ymin = auprc_lo, ymax = auprc_hi),
                          width = 0.3, colour = "grey40") +
            coord_flip() +
            coord_cartesian(ylim = c(0.2, 1.0)) +
            labs(x = NULL, y = "AUPRC") +
            theme_minimal(base_size = 13)
    })

    output$score_table <- renderTable({
        d <- scores[, c("model", "auroc", "auprc", "within_gene")]
        d$auroc <- sprintf("%.3f", d$auroc)
        d$auprc <- sprintf("%.3f", d$auprc)
        d$within_gene <- sprintf("%.3f", d$within_gene)
        names(d) <- c("model", "AUROC", "AUPRC", "within-gene AUROC")
        d
    }, rownames = FALSE)

    output$roc <- renderImage({
        list(src = "reports/figures/04_roc_pr.png",
             width = "100%", contentType = "image/png")
    }, deleteFile = FALSE)

    output$calibration <- renderImage({
        list(src = "reports/figures/05_calibration.png",
             width = "100%", contentType = "image/png")
    }, deleteFile = FALSE)

    output$splits <- renderTable(splits, rownames = FALSE)
}

shinyApp(ui, server)
