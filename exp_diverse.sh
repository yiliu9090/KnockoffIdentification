#!/usr/bin/env bash
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error
# prepare folders
exp_path=exp_diverse
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path
source_models="Llama-3-70B GPT-3-Turbo Gemini-1.5-Pro GPT-4o"
datasets="AcademicResearch EducationMaterial FoodCusine MedicalText ProductReview TravelTourism ArtCulture Entertainment GovernmentPublic NewsArticle Religious Business Environmental LegalDocument OnlineContent Sports Code Finance LiteratureCreativeWriting PersonalCommunication TechnicalWriting"
settings='gemma-9b:gemma-9b-instruct'
scoring_models="gemma-9b-instruct"
rewrite_model="gemma-9b-instruct"
gpu_device='cuda'
data_split_2="AcademicResearch EducationMaterial FoodCusine MedicalText ProductReview TravelTourism ArtCulture Entertainment GovernmentPublic NewsArticle"
data_split_1="Religious Business Environmental LegalDocument OnlineContent Sports Code Finance LiteratureCreativeWriting PersonalCommunication TechnicalWriting"
train_model_1="Llama-3-70B"
eval_models_1="GPT-3-Turbo GPT-4o Gemini-1.5-Pro"
train_model_2="GPT-3-Turbo"
eval_models_2="Llama-3-70B"

# evaluate ImBD
trained_model_path=scripts/ImBD/ckpt/ai_detection_500_spo_lr_0.0001_beta_0.05_a_1
for setting in 1 2; do
  if [ "$setting" -eq 1 ]; then
    train_dataset=$data_split_1
    eval_datasets=$data_split_2
    train_model=$train_model_1
    eval_models=$eval_models_1
  else
    train_dataset=$data_split_2
    eval_datasets=$data_split_1
    train_model=$train_model_2
    eval_models=$eval_models_2
  fi

  my_train_dataset_str=""
  for D1 in $train_dataset; do
    if [ "$D1" = "Code" ] || [ "$D1" = "LiteratureCreativeWriting" ]; then
      echo "Skipping dataset: $D1"
      continue
    fi
  
    if [ -z "$my_train_dataset_str" ]; then
      my_train_dataset_str="${data_path}/${D1}_${train_model}"
    else
      my_train_dataset_str="${my_train_dataset_str}&${data_path}/${D1}_${train_model}"
    fi
  done
  echo "Train data: ---------------------------------- "
  for D in $train_dataset; do
    for M in $train_model; do
      if [ ! -f "$res_path/${D}_${M}.imbd.json" ]; then
        echo "${setting}Evaluating ImBD on ${D}_${M} ..."
        python scripts/detect_ImBD.py --eval_only --base_model "$scoring_models" --eval_dataset "$data_path/${D}_${M}" --output_file "$res_path/${D}_${M}" --from_pretrained "$trained_model_path" --device $gpu_device
      else
        echo "${setting}:Skipping ${D}_${M} (imbd file already exists)"
      fi
    done
  done
  
  echo "Eval datasets: ----------------------------------"
  for D in $eval_datasets; do
    for M in $eval_models; do
      if [ ! -f "$res_path/${D}_${M}.imbd.json" ]; then
        echo "${setting}Evaluating ImBD on ${D}_${M} ..."
        python scripts/detect_ImBD.py --eval_only --base_model "$scoring_models" --eval_dataset "$data_path/${D}_${M}" --output_file "$res_path/${D}_${M}" --from_pretrained "$trained_model_path" --device $gpu_device
      else
        echo "${setting}:Skipping ${D}_${M} (imbd file already exists)"
      fi
    done
  done
done
for D in $datasets; do
  for M in $source_models; do
    if [ ! -f "$res_path/${D}_${M}.imbd.json" ]; then
      echo "${setting}Evaluating ImBD on ${D}_${M} ..."
      python scripts/detect_ImBD.py --eval_only --base_model "$scoring_models" --eval_dataset "$data_path/${D}_${M}" --output_file "$res_path/${D}_${M}" --from_pretrained "$trained_model_path" --device $gpu_device
    else
      echo "${setting}:Skipping ${D}_${M} (imbd file already exists)"
    fi
  done
done
#evaluate L2D (using pretrained mamba413/L2D from HuggingFace)
l2d_model_path=mamba413/L2D
for D in $datasets; do
  for M in $source_models; do
    echo "$(date), Evaluating L2D on ${D}_${M} ..."
    python scripts/detect_l2d.py \
      --eval_only \
      --base_model    "$scoring_models" \
      --rewrite_model "$rewrite_model" \
      --eval_dataset  "$data_path/${D}_${M}" \
      --output_file   "$res_path/${D}_${M}" \
      --from_pretrained "$l2d_model_path" \
      --device        "$gpu_device"
  done
done