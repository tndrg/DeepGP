#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run GWAS and LD clumping for DeepGP SNP selection.

Required:
  --pfile PREFIX          PLINK 2 pfile prefix for QC'ed imputed genotype data.
  --pheno FILE           Phenotype file with FID, IID, and phenotype column.
  --pheno-name NAME      Phenotype column name, for example T2D.
  --keep FILE            European training-set sample IDs, two columns: FID IID.
  --out-dir DIR          Output directory.

Optional:
  --covar FILE           Optional covariate file with FID, IID, sex, age, and PCs.
  --covar-name LIST      Comma-separated covariate columns.
                         Default: sex,age,PC1,PC2,PC3,PC4,PC5,PC6
  --plink2 BIN           PLINK 2 executable. Default: plink2
  --threads N            Number of CPU threads. Default: 8
  --memory MB            Memory limit passed to PLINK. Default: 32000
  --glm-extra STRING     Extra arguments appended to --glm.
                         Default: firth-fallback cols=+a1freq
  --clump-field NAME     GWAS p-value column used for clumping. Default: P
  --clump-p1 VALUE       Index SNP p-value threshold. Default: 0.05
  --clump-p2 VALUE       Secondary SNP p-value threshold. Default: 0.05
  --clump-r2 VALUE       LD r2 threshold. Default: 0.7
  --clump-kb VALUE       LD window in kb. Default: 500

Example:
  bash scripts/gwas_plink.sh \
    --pfile /path/to/ukb_qc_autosomes \
    --pheno data/phenotypes/cmd_phenotypes.tsv \
    --pheno-name T2D \
    --keep data/splits/T2D_eur_train.keep \
    --out-dir results/gwas/T2D
EOF
}

PFILE=""
PHENO=""
PHENO_NAME=""
COVAR=""
KEEP=""
OUT_DIR=""
COVAR_NAME="sex,age,PC1,PC2,PC3,PC4,PC5,PC6"
PLINK2="plink2"
THREADS=8
MEMORY=32000
CLUMP_FIELD="P"
GLM_EXTRA=("firth-fallback" "cols=+a1freq")
CLUMP_P1=0.05
CLUMP_P2=0.05
CLUMP_R2=0.7
CLUMP_KB=500

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pfile) PFILE="$2"; shift 2 ;;
    --pheno) PHENO="$2"; shift 2 ;;
    --pheno-name) PHENO_NAME="$2"; shift 2 ;;
    --covar) COVAR="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --covar-name) COVAR_NAME="$2"; shift 2 ;;
    --plink2) PLINK2="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --memory) MEMORY="$2"; shift 2 ;;
    --glm-extra) read -r -a GLM_EXTRA <<< "$2"; shift 2 ;;
    --clump-field) CLUMP_FIELD="$2"; shift 2 ;;
    --clump-p1) CLUMP_P1="$2"; shift 2 ;;
    --clump-p2) CLUMP_P2="$2"; shift 2 ;;
    --clump-r2) CLUMP_R2="$2"; shift 2 ;;
    --clump-kb) CLUMP_KB="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

for value in PFILE PHENO PHENO_NAME KEEP OUT_DIR; do
  if [[ -z "${!value}" ]]; then
    echo "Missing required argument: --${value,,}" >&2
    usage
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

GWAS_PREFIX="$OUT_DIR/${PHENO_NAME}.eur_train.gwas"
CLUMP_PREFIX="$OUT_DIR/${PHENO_NAME}.eur_train.ld_clumped"

GWAS_COVAR_ARGS=()
if [[ -n "$COVAR" ]]; then
  GWAS_COVAR_ARGS=(--covar "$COVAR" --covar-name "$COVAR_NAME")
fi

"$PLINK2" \
  --pfile "$PFILE" \
  --keep "$KEEP" \
  --pheno "$PHENO" \
  --pheno-name "$PHENO_NAME" \
  "${GWAS_COVAR_ARGS[@]}" \
  --glm hide-covar "${GLM_EXTRA[@]}" \
  --threads "$THREADS" \
  --memory "$MEMORY" \
  --out "$GWAS_PREFIX"

GWAS_FILE="$(find "$OUT_DIR" -maxdepth 1 -type f -name "${PHENO_NAME}.eur_train.gwas.*glm*" | head -n 1)"
if [[ -z "$GWAS_FILE" ]]; then
  echo "GWAS finished, but no .glm output was found under $OUT_DIR" >&2
  exit 1
fi

"$PLINK2" \
  --pfile "$PFILE" \
  --keep "$KEEP" \
  --clump "$GWAS_FILE" \
  --clump-p-field "$CLUMP_FIELD" \
  --clump-id-field ID \
  --clump-p1 "$CLUMP_P1" \
  --clump-p2 "$CLUMP_P2" \
  --clump-r2 "$CLUMP_R2" \
  --clump-kb "$CLUMP_KB" \
  --threads "$THREADS" \
  --memory "$MEMORY" \
  --out "$CLUMP_PREFIX"

awk '
  NR == 1 {
    for (i = 1; i <= NF; i++) {
      if ($i == "ID" || $i == "SNP") id_col = i
    }
    next
  }
  id_col && $id_col != "NONE" { print $id_col }
' "${CLUMP_PREFIX}.clumps" > "${CLUMP_PREFIX}.snplist"

cat <<EOF
GWAS output: ${GWAS_FILE}
LD clumps:   ${CLUMP_PREFIX}.clumps
Tag SNPs:    ${CLUMP_PREFIX}.snplist
EOF
