#!/usr/bin/env Rscript

# ============================================================
# Multi-Gene Mutation + Expression Panels
# With FDR Correction (Benjamini-Hochberg)
# ============================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(tidyr)
})

# ----------------
# Gene list
# ----------------
genes <- c(
  "C1QBP","CBX5","CCR5","CDK1","CDK2","CDK7","COQ9","CRKL",
  "CUL7","DDB1","DHX15","DSN1","EDC4","EGFR","EPS15","EVL",
  "HDAC1","HDAC2","HDAC3","HDAC4","HEY2","HINT2","IRS4",
  "MCM6","MRM3","PLK1","RBBP4","RBM39","SFSA2","SIN3A",
  "SKP1","SKP2","SMAT4","SP100","SRRM2","STAT1",
  "SUMO1","TFAP4","TP53"
)

expression_file <- "data/TCGA-PAN.RNAseq_expression.tsv"
output_dir <- "plots"

if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

expression <- read_tsv(expression_file, show_col_types = FALSE)

deleterious_colors <- c(
  "FALSE" = "#E41A1C",
  "TRUE"  = "#377EB8",
  "NA"    = "grey80"
)

# ============================================================
# First pass: compute ALL p-values
# ============================================================

results_table <- data.frame(
  Gene = character(),
  Wilcoxon_p = numeric(),
  Kruskal_p = numeric(),
  stringsAsFactors = FALSE
)

for (gene in genes) {

  mutation_file <- paste0("data/TCGA-PAN.mutation_", tolower(gene), ".tsv")
  if (!file.exists(mutation_file)) next
  if (!(gene %in% colnames(expression))) next

  mutations <- read_tsv(mutation_file, show_col_types = FALSE)
  gene_mutations <- mutations %>% filter(Gene == gene)

  sample_status <- gene_mutations %>%
    mutate(Mutation_Status = "MT") %>%
    select(Sample, Mutation_Status, Deleterious, Variant_Classification) %>%
    distinct()

  expression_status <- expression %>%
    select(Sample, all_of(gene)) %>%
    left_join(sample_status, by = "Sample") %>%
    mutate(
      Mutation_Status        = ifelse(is.na(Mutation_Status), "WT", Mutation_Status),
      Deleterious            = ifelse(is.na(Deleterious), "NA", Deleterious),
      Variant_Classification = ifelse(is.na(Variant_Classification), "WT", Variant_Classification)
    )

  # ---------------- Wilcoxon ----------------
  wilcox_p <- NA
  if (length(unique(expression_status$Mutation_Status)) == 2) {
    test_data <- expression_status %>%
      filter(!is.na(.data[[gene]]))
    wilcox_p <- tryCatch(
      wilcox.test(test_data[[gene]] ~ test_data$Mutation_Status)$p.value,
      error = function(e) NA
    )
  }

  # ---------------- Kruskal ----------------
  kruskal_p <- NA
  if (length(unique(expression_status$Variant_Classification)) > 1) {
    test_data <- expression_status %>%
      filter(!is.na(.data[[gene]]))
    kruskal_p <- tryCatch(
      kruskal.test(test_data[[gene]] ~ test_data$Variant_Classification)$p.value,
      error = function(e) NA
    )
  }

  results_table <- rbind(
    results_table,
    data.frame(
      Gene = gene,
      Wilcoxon_p = wilcox_p,
      Kruskal_p = kruskal_p
    )
  )
}

# ============================================================
# FDR correction (Benjamini-Hochberg)
# ============================================================

results_table$Wilcoxon_FDR <- p.adjust(results_table$Wilcoxon_p, method = "BH")
results_table$Kruskal_FDR  <- p.adjust(results_table$Kruskal_p, method = "BH")

# Save summary table
write.csv(results_table,
          file = file.path(output_dir, "Gene_pvalues_with_FDR.csv"),
          row.names = FALSE)

# ============================================================
# Second pass: generate plots using FDR-adjusted p-values
# ============================================================

for (i in 1:nrow(results_table)) {

  gene <- results_table$Gene[i]
  message("Plotting ", gene)

  mutation_file <- paste0("data/TCGA-PAN.mutation_", tolower(gene), ".tsv")
  if (!file.exists(mutation_file)) next

  mutations <- read_tsv(mutation_file, show_col_types = FALSE)
  gene_mutations <- mutations %>% filter(Gene == gene)

  sample_status <- gene_mutations %>%
    mutate(Mutation_Status = "MT") %>%
    select(Sample, Mutation_Status, Deleterious, Variant_Classification) %>%
    distinct()

  expression_status <- expression %>%
    select(Sample, all_of(gene)) %>%
    left_join(sample_status, by = "Sample") %>%
    mutate(
      Mutation_Status        = ifelse(is.na(Mutation_Status), "WT", Mutation_Status),
      Deleterious            = ifelse(is.na(Deleterious), "NA", Deleterious),
      Variant_Classification = ifelse(is.na(Variant_Classification), "WT", Variant_Classification)
    )

  ymax <- max(expression_status[[gene]], na.rm = TRUE)

  # ================= Panel C =================
  pC <- ggplot(expression_status,
               aes(x = Mutation_Status,
                   y = .data[[gene]],
                   fill = Deleterious)) +
    geom_violin(trim = FALSE, alpha = 0.6, na.rm = TRUE) +
    geom_boxplot(width = 0.15, outlier.size = 0.5,
                 position = position_dodge(width = 0.9),
                 na.rm = TRUE) +
    scale_fill_manual(values = deleterious_colors) +
    labs(x = "", y = "Expression (log2 TPM)") +
    theme_classic(base_size = 16)

  if (!is.na(results_table$Wilcoxon_FDR[i])) {
    pC <- pC +
      annotate("text",
               x = 1.5,
               y = ymax * 1.08,
               label = paste0("FDR = ",
                              format(results_table$Wilcoxon_FDR[i],
                                     digits = 3, scientific = TRUE)),
               size = 5)
  }
  
  # ================= Panel D =================

  # Filter only for plotting (do NOT change statistical results)
  expression_status_D <- expression_status %>%
    group_by(Variant_Classification) %>%
    filter(n() >= 2) %>%
    ungroup()

  pD <- ggplot(expression_status_D,
              aes(x = Variant_Classification,
                  y = .data[[gene]],
                  fill = Variant_Classification)) +
    geom_violin(trim = FALSE, alpha = 0.5, na.rm = TRUE) +
    geom_boxplot(width = 0.15, outlier.size = 0.5,
                position = position_dodge(width = 0.9),
                na.rm = TRUE) +
    labs(x = "", y = "Expression (log2 TPM)") +
    theme_bw(base_size = 16) +
    theme(
      axis.title.y = element_text(size = 20),
      axis.text = element_text(size = 18),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "none"
    )

  # Keep original FDR (do NOT recompute)
  if (!is.na(results_table$Kruskal_FDR[i])) {
    ymax_D <- max(expression_status_D[[gene]], na.rm = TRUE)
    pD <- pD +
      annotate("text",
              x = mean(seq_along(unique(expression_status_D$Variant_Classification))),
              y = ymax_D * 1.08,
              label = paste0("FDR = ",
                              format(results_table$Kruskal_FDR[i],
                                    digits = 3, scientific = TRUE)),
              size = 5)
  }

  # # ================= Panel D =================
  # pD <- ggplot(expression_status,
  #              aes(x = Variant_Classification,
  #                  y = .data[[gene]],
  #                  fill = Variant_Classification)) +
  #   geom_violin(trim = FALSE, alpha = 0.5, na.rm = TRUE) +
  #   geom_boxplot(width = 0.15, outlier.size = 0.5,
  #                position = position_dodge(width = 0.9),
  #                na.rm = TRUE) +
  #   labs(x = "", y = "Expression (log2 TPM)") +
  #   theme_bw(base_size = 16) +
  #   theme(
  #     axis.title.y = element_text(size = 20),
  #     axis.text = element_text(size = 18),
  #     axis.text.x = element_text(angle = 45, hjust = 1),
  #     legend.position = "none"
  #   )

  # if (!is.na(results_table$Kruskal_FDR[i])) {
  #   pD <- pD +
  #     annotate("text",
  #              x = mean(1:length(unique(expression_status$Variant_Classification))),
  #              y = ymax * 1.08,
  #              label = paste0("FDR = ",
  #                             format(results_table$Kruskal_FDR[i],
  #                                    digits = 3, scientific = TRUE)),
  #              size = 5)
  # }
  # ggsave(file.path(output_dir, paste0("PanelC_", gene, ".png")),
  #        pC, width = 6, height = 5, dpi = 300)

  ggsave(file.path(output_dir, paste0("PanelD_", gene, ".png")),
         pD, width = 8, height = 5, dpi = 300)

  # ggsave(file.path(output_dir, paste0("PanelC_", gene, ".pdf")),
  #        pC, width = 6, height = 5)

  # ggsave(file.path(output_dir, paste0("PanelD_", gene, ".pdf")),
  #        pD, width = 8, height = 5)
}

message("All plots completed with BH-FDR correction.")
