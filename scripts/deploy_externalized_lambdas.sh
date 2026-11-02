#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy_externalized_lambdas.sh \
    --stack-name <stack-name> \
    --artifact-bucket <s3-bucket> \
    --version <artifact-version> \
    [--region us-east-1] \
    [--prefix lambda-artifacts] \
    [--profile <aws-profile>]

Example:
  scripts/deploy_externalized_lambdas.sh \
    --stack-name amc-acm-custom-model-stack-uk \
    --artifact-bucket my-artifacts-bucket \
    --version 2026-11-02-1 \
    --region us-east-1 \
    --prefix lambda-artifacts
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

STACK_NAME=""
ARTIFACT_BUCKET=""
VERSION=""
REGION="us-east-1"
PREFIX="lambda-artifacts"
PROFILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stack-name)
      STACK_NAME="$2"; shift 2;;
    --artifact-bucket)
      ARTIFACT_BUCKET="$2"; shift 2;;
    --version)
      VERSION="$2"; shift 2;;
    --region)
      REGION="$2"; shift 2;;
    --prefix)
      PREFIX="$2"; shift 2;;
    --profile)
      PROFILE="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1;;
  esac
done

if [[ -z "$STACK_NAME" || -z "$ARTIFACT_BUCKET" || -z "$VERSION" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

require_cmd aws
require_cmd zip

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMPLATE_PATH="${REPO_ROOT}/AMC_Custom_Models_Accelerator.yaml"

if [[ ! -f "$TEMPLATE_PATH" ]]; then
  echo "Template not found: $TEMPLATE_PATH" >&2
  exit 1
fi

LAMBDA_ROOT="${REPO_ROOT}/lambdas"
if [[ ! -d "$LAMBDA_ROOT" ]]; then
  echo "Lambda source directory not found: $LAMBDA_ROOT" >&2
  exit 1
fi

mapfile -t LAMBDAS < <(find "$LAMBDA_ROOT" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort)
if [[ ${#LAMBDAS[@]} -eq 0 ]]; then
  echo "No lambda directories found under: $LAMBDA_ROOT" >&2
  exit 1
fi

AWS_ARGS=(--region "$REGION")
if [[ -n "$PROFILE" ]]; then
  AWS_ARGS+=(--profile "$PROFILE")
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Packaging and uploading Lambda artifacts..."
for name in "${LAMBDAS[@]}"; do
  src="${REPO_ROOT}/lambdas/${name}/index.py"
  if [[ ! -f "$src" ]]; then
    echo "Missing Lambda source file: $src" >&2
    exit 1
  fi

  work_dir="${TMP_DIR}/${name}"
  mkdir -p "$work_dir"
  cp "$src" "${work_dir}/index.py"

  zip_path="${TMP_DIR}/${name}.zip"
  (
    cd "$work_dir"
    zip -q -r "$zip_path" index.py
  )

  s3_key="${PREFIX}/${name}/${VERSION}.zip"
  aws "${AWS_ARGS[@]}" s3 cp "$zip_path" "s3://${ARTIFACT_BUCKET}/${s3_key}"
  echo "Uploaded: s3://${ARTIFACT_BUCKET}/${s3_key}"
done

echo "Deploying CloudFormation stack update..."
aws "${AWS_ARGS[@]}" cloudformation deploy \
  --template-file "$TEMPLATE_PATH" \
  --stack-name "$STACK_NAME" \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    LambdaCodeS3Bucket="${ARTIFACT_BUCKET}" \
    LambdaCodeS3KeyPrefix="${PREFIX}" \
    LambdaCodeVersion="${VERSION}"

echo "Done."
